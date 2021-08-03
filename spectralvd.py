import pickle
import random

import numpy as np
from tensorflow import keras

from nils.reconstruction_module import cleanup_formfactor
from nils.simulate_spectrometer_signal import get_crisp_signal


class SpectralVD:

    def __init__(self):
        self.load_formfactors()

        self.model = keras.models.load_model("models/both_model")
        with open("models/both_scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)

    def load_formfactors(self):
        with open("ocelot80k.pkl", "rb") as file:
            data = pickle.load(file)
        
        currents = [(sample["s"][:1000], sample["I"][:1000]) for sample in data]
        filtered = [(s, current) for s, current in currents if current.max() > 1000]
        
        def current2formfactor(s, current, grating="both"):
            """Convert a current to its corresponding cleaned form factor."""
            frequency, formfactor, formfactor_noise, detlim = get_crisp_signal(s, current, n_shots=10, which_set=grating)
            clean_frequency, clean_formfactor, _ = cleanup_formfactor(frequency, formfactor, formfactor_noise, detlim, channels_to_remove=[])

            return clean_frequency, clean_formfactor

        formfactors_both = [current2formfactor(*current, grating="both") for current in random.choices(filtered, k=10)]
        self.formfactors = formfactors_both
    
    def read_crisp(self):
        i = np.random.randint(0, len(self.formfactors))
        return self.formfactors[i]
    
    def ann_reconstruction(self, crisp):
        crisp = crisp[1].reshape((1,-1))
        X_scaled = self.scalers["X"].transform(crisp)
        y_scaled = self.model.predict(X_scaled)
        y = y_scaled / self.scalers["y"]

        limit = 0.00020095917745111108
        s = np.linspace(-limit, limit, 100)

        return s, y.squeeze()

    def nils_reconstruction(self, crisp):
        limit = 0.00020095917745111108
        s = np.linspace(-limit, limit, 100)

        return s, np.random.random(100) * 6e3
