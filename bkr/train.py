import json
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from nils.reconstruction_module import cleanup_formfactor
from nils.simulate_spectrometer_signal import get_crisp_signal


print("Loading ...")
with open("../research/ocelot80k.pkl", "rb") as file:
    data = pickle.load(file)

currents = [(sample["s"][:1000], sample["I"][:1000]) for sample in data]
del data

print("Preprocessing ...")
# Only keep samples with current profiles that exceed 1 kA
filtered = [(s, current) for s, current in currents if current.max() > 1000]
len(filtered)

# Shift center of mass of each current profile onto the origin

def shift_onto_center_of_mass(s, current):
    """Shift a current profile such that its center of mass it at 0."""
    cm = (s * current).sum() / current.sum()
    return s - cm, current

shifted = [shift_onto_center_of_mass(s, current) for s, current in filtered]

# Interpolate all current profiles onto the same samples of s

limit = max(max(np.abs(s)) for s, _ in shifted)
new_s = np.linspace(-limit, limit, 100)

interpolated = [(new_s, np.interp(new_s, s, current, left=0, right=0)) for s, current in shifted]

print("Making formfactors ...")
# Make formfactors
def current2formfactor(s, current, grating="both"):
    """Convert a current to its corresponding cleaned form factor."""
    frequency, formfactor, formfactor_noise, detlim = get_crisp_signal(s, current, n_shots=10, which_set=grating)
    clean_frequency, clean_formfactor, _ = cleanup_formfactor(frequency, formfactor, formfactor_noise, detlim, channels_to_remove=[])

    return clean_frequency, clean_formfactor

formfactors = [current2formfactor(*current, grating="both") for current in filtered]

print("Training ...")

def train(formfactors, currents, epochs=500):
    """Train and return model to infer currents from formfactors."""
    X = np.stack([formfactor for _, formfactor in formfactors])
    y = np.stack([current for _, current in interpolated])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    X_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    y_scaler = 1 / y_train.max(axis=1).mean()
    y_train_scaled = y_train * y_scaler
    
    model = keras.Sequential([
        layers.Dense(200, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(y_train.shape[1], activation="relu")]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    history = model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=64, validation_split=0.25)
    
    # plot_history(history)
    # evaluate_model(model, X_test, y_test, X_scaler, y_scaler)
    # do_example_predictions(model, X_test, y_test, X_scaler, y_scaler)
    
    return model, X_scaler, y_scaler

model, X_scaler, y_scaler = train(formfactors, interpolated, epochs=200)

model.save("model")
scaler_params = {
    "X_min": list(X_scaler.data_min_),
    "X_max": list(X_scaler.data_max_),
    "y_scaler": y_scaler
}
with open("scalers.json", "w") as f:
    json.dump(scaler_params, f)

print("Done!")
