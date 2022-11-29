from pathlib import Path
import pickle

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from nils.reconstruction_module import cleanup_formfactor
from nils.reconstruction_module_after_diss import master_recon


def from_pickle(path):
    with open(f"{path}.pkl", "rb") as f:
        obj = pickle.load(f)
    
    return obj


def to_pickle(obj, path):
    with open(f"{path}.pkl", "wb") as f:
        pickle.dump(obj, f)

        
def compute_max_left(current):
    revcurrent = np.flip(current)
    return np.flip(np.array([revcurrent[i:].max() for i in range(len(current))]))


def compute_max_right(current):
    return np.array([current[i:].max() for i in range(len(current))])


def find_edges(s, current, threshold=0.01):
    max_left = compute_max_left(current)
    left_idx = np.where(max_left >= threshold * max_left.max())[0][0]
    left = s[left_idx]
    
    max_right = compute_max_right(current)
    right_idx = np.where(max_right >= threshold * max_right.max())[0][-1]
    right = s[right_idx]

    return left, right
    
    
def resample(s, current, left, right, n):
    new_s = np.linspace(left, right, n)
    new_current = np.interp(new_s, s, current, left=0, right=0)
    return new_s, new_current
    
    
def center_on_zero(s, current, left, right):
    old_center = left + ((right - left) / 2)
    new_s = s - old_center
    return new_s, current


class Fixed:
    
    def __init__(self, width=140e-6, n_samples=100):
        self.width = width
        self.n_samples = n_samples
        
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        self.model = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(n_samples, activation="relu")]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    def fit(self, formfactors, currents, epochs=1000, verbose=1):
        X = self._preprocess_formfactors(formfactors)
        y = self._preprocess_currents(currents)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        self.history = self.model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=64, verbose=verbose)
            
    def predict(self, formfactors):
        X = self._preprocess_formfactors(formfactors)
        X_scaled = self.X_scaler.transform(X)
        y_scaled = self.model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)
        s = np.linspace(-self.width/2, self.width/2, self.n_samples)
        return np.array([(s, yi) for yi in y])
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model = keras.models.load_model(f"{path}/model")
        svd.X_scaler = from_pickle(f"{path}/X_scaler")
        svd.y_scaler = from_pickle(f"{path}/y_scaler")
        svd.n_samples = from_pickle(f"{path}/n_samples")
        svd.width = from_pickle(f"{path}/width")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        self.model.save(f"{path}/model")
        to_pickle(self.X_scaler, f"{path}/X_scaler")
        to_pickle(self.y_scaler, f"{path}/y_scaler")
        to_pickle(self.n_samples, f"{path}/n_samples")
        to_pickle(self.width, f"{path}/width")
    
    def _preprocess_formfactors(self, formfactors):
        X = np.stack([formfactor for _, formfactor in formfactors])
        return X
    
    def _preprocess_currents(self, currents):
        shifted = [self._shift_onto_center_of_mass(s, current) for s, current in currents]
        
        new_s = np.linspace(-self.width/2, self.width/2, self.n_samples)
        interpolated = [(new_s, np.interp(new_s, s, current, left=0, right=0)) for s, current in shifted]
        
        y = np.stack([current for _, current in interpolated])
        
        return y
    
    def _shift_onto_center_of_mass(self, s, current):
        """Shift a current profile such that its center of mass it at 0."""
        cm = (s * current).sum() / current.sum()
        return s - cm, current
    

class Adaptive:
    
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
    
    def _preprocess_rf(self, rf):
        X = np.stack(rf)
        return X
        
    def _preprocess_formfactors(self, formfactors):
        X = np.stack([formfactor for _, formfactor in formfactors])
        return X
    
    def _preprocess_currents(self, currents):
        edges = np.array([find_edges(s, current) for s, current in currents])
        widths = edges[:,1] - edges[:,0]
        y1 = widths.reshape(-1, 1)
        
        resampled = [resample(s, current, left, right, self.n_samples) for (s, current), (left, right) in zip(currents, edges)]
        centered = [center_on_zero(s, current, left, right) for (s, current), (left, right) in zip(resampled, edges)]
        y2 = np.stack([current for _, current in centered])
        
        return y1, y2
    
    
class AdaptiveANN(Adaptive):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
        self.X_scaler = MinMaxScaler()
        self.y1_scaler = MinMaxScaler()
        self.y2_scaler = MinMaxScaler()
        
        self.model1 = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(1, activation="relu")]
        )
        self.model1.compile(optimizer="adam", loss="mse", metrics=["mae"])
        
        self.model2 = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(self.n_samples, activation="relu")]
        )
        self.model2.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model1 = keras.models.load_model(f"{path}/model1")
        svd.model2 = keras.models.load_model(f"{path}/model2")
        svd.X_scaler = from_pickle(f"{path}/X_scaler")
        svd.y1_scaler = from_pickle(f"{path}/y1_scaler")
        svd.y2_scaler = from_pickle(f"{path}/y2_scaler")
        svd.n_samples = from_pickle(f"{path}/n_samples")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        self.model1.save(f"{path}/model1")
        self.model2.save(f"{path}/model2")
        to_pickle(self.X_scaler, f"{path}/X_scaler")
        to_pickle(self.y1_scaler, f"{path}/y1_scaler")
        to_pickle(self.y2_scaler, f"{path}/y2_scaler")
        to_pickle(self.n_samples, f"{path}/n_samples")


class AdaptiveANNRF(AdaptiveANN):
    
    def fit(self, rf, currents, epochs=1000, verbose=1):
        X = self._preprocess_rf(rf)
        y1, y2 = self._preprocess_currents(currents)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y1_scaled = self.y1_scaler.fit_transform(y1)
        y2_scaled = self.y2_scaler.fit_transform(y2)
        
        self.history1 = self.model1.fit(X_scaled, y1_scaled, epochs=epochs, batch_size=64, verbose=verbose)
        self.history2 = self.model2.fit(X_scaled, y2_scaled, epochs=epochs, batch_size=64, verbose=verbose)
    
    def predict(self, rf):
        X = self._preprocess_rf(rf)
        X_scaled = self.X_scaler.transform(X)
        
        y1_scaled = self.model1.predict(X_scaled)
        y1 = self.y1_scaler.inverse_transform(y1_scaled)
        widths = y1.squeeze()
        s = [np.linspace(-w/2, w/2, self.n_samples) for w in widths]
        
        y2_scaled = self.model2.predict(X_scaled)
        y2 = self.y2_scaler.inverse_transform(y2_scaled)

        return np.array(list(zip(s, y2)))


class AdaptiveANNTHz(AdaptiveANN):
    
    def fit(self, formfactors, currents, epochs=1000, verbose=1):
        X = self._preprocess_formfactors(formfactors)
        y1, y2 = self._preprocess_currents(currents)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y1_scaled = self.y1_scaler.fit_transform(y1)
        y2_scaled = self.y2_scaler.fit_transform(y2)
        
        self.history1 = self.model1.fit(X_scaled, y1_scaled, epochs=epochs, batch_size=64, verbose=verbose)
        self.history2 = self.model2.fit(X_scaled, y2_scaled, epochs=epochs, batch_size=64, verbose=verbose)
            
    def predict(self, formfactors):
        X = self._preprocess_formfactors(formfactors)
        X_scaled = self.X_scaler.transform(X)
        
        y1_scaled = self.model1.predict(X_scaled)
        y1 = self.y1_scaler.inverse_transform(y1_scaled)
        widths = y1.squeeze()
        s = [np.linspace(-w/2, w/2, self.n_samples) for w in widths]
        
        y2_scaled = self.model2.predict(X_scaled)
        y2 = self.y2_scaler.inverse_transform(y2_scaled)

        return np.array(list(zip(s, y2)))
    

class AdaptiveANNRFTHz(AdaptiveANN):
    
    def fit(self, rf, formfactors, currents, epochs=1000, verbose=1):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        y1, y2 = self._preprocess_currents(currents)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y1_scaled = self.y1_scaler.fit_transform(y1)
        y2_scaled = self.y2_scaler.fit_transform(y2)
        
        self.history1 = self.model1.fit(X_scaled, y1_scaled, epochs=epochs, batch_size=64, verbose=verbose)
        self.history2 = self.model2.fit(X_scaled, y2_scaled, epochs=epochs, batch_size=64, verbose=verbose)
            
    def predict(self, rf, formfactors):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        X_scaled = self.X_scaler.transform(X)
        
        y1_scaled = self.model1.predict(X_scaled)
        y1 = self.y1_scaler.inverse_transform(y1_scaled)
        widths = y1.squeeze()
        s = [np.linspace(-w/2, w/2, self.n_samples) for w in widths]
        
        y2_scaled = self.model2.predict(X_scaled)
        y2 = self.y2_scaler.inverse_transform(y2_scaled)

        return np.array(list(zip(s, y2)))


class AdaptiveKNN(Adaptive):
    
    def __init__(self, n_neighbors=2, weights="distance", **kwargs):
        super().__init__(**kwargs)
        
        self.model1 = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        self.model2 = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model1 = from_pickle(f"{path}/model1")
        svd.model2 = from_pickle(f"{path}/model2")
        svd.n_samples = from_pickle(f"{path}/n_samples")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        to_pickle(self.model1, f"{path}/model1")
        to_pickle(self.model2, f"{path}/model2")
        to_pickle(self.n_samples, f"{path}/n_samples")
    

class AdaptiveKNNRF(AdaptiveKNN):
    
    def fit(self, rf, currents):
        X = self._preprocess_rf(rf)
        y1, y2 = self._preprocess_currents(currents)
        
        self.model1.fit(X, y1)
        self.model2.fit(X, y2)
    
    def predict(self, rf):
        X = self._preprocess_rf(rf)
        
        y1 = self.model1.predict(X)
        widths = y1.squeeze()
        s = [np.linspace(-w/2, w/2, self.n_samples) for w in widths]
        
        y2 = self.model2.predict(X)
        
        return np.array(list(zip(s, y2)))


class AdaptiveKNNTHz(AdaptiveKNN):
    
    def fit(self, formfactors, currents):
        X = self._preprocess_formfactors(formfactors)
        y1, y2 = self._preprocess_currents(currents)
        
        self.model1.fit(X, y1)
        self.model2.fit(X, y2)
    
    def predict(self, formfactors):
        X = self._preprocess_formfactors(formfactors)
        
        y1 = self.model1.predict(X)
        widths = y1.squeeze()
        s = [np.linspace(-w/2, w/2, self.n_samples) for w in widths]
        
        y2 = self.model2.predict(X)
        
        return np.array(list(zip(s, y2)))


class AdaptiveKNNRFTHz(AdaptiveKNN):
    
    def fit(self, rf, formfactors, currents):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        y1, y2 = self._preprocess_currents(currents)
        
        self.model1.fit(X, y1)
        self.model2.fit(X, y2)
    
    def predict(self, rf, formfactors):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        
        y1 = self.model1.predict(X)
        widths = y1.squeeze()
        s = [np.linspace(-w/2, w/2, self.n_samples) for w in widths]
        
        y2 = self.model2.predict(X)
        
        return np.array(list(zip(s, y2)))

    
class ReverseRF:
    
    def __init__(self, n_rf=13):
        self.n_rf = n_rf
    
    def _preprocess_formfactors(self, formfactors):
        X = np.stack([formfactor for _, formfactor in formfactors])
        return X
    
    def _preprocess_rf(self, rf):
        X = np.stack(rf)
        return X
    
    
class ReverseRFANN(ReverseRF):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.X_scaler = MinMaxScaler()
        self.y_scaler = StandardScaler()
        
        self.model = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(self.n_rf, activation=None)]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    def fit(self, formfactors, rf, epochs=1000, verbose=1):
        X = self._preprocess_formfactors(formfactors)
        y = self._preprocess_rf(rf)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        history = self.model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=64, verbose=verbose)
        
        self.history = history.history
            
    def predict(self, formfactors):
        X = self._preprocess_formfactors(formfactors)
        X_scaled = self.X_scaler.transform(X)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        return y
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model = keras.models.load_model(f"{path}/model")
        svd.X_scaler = from_pickle(f"{path}/X_scaler")
        svd.y_scaler = from_pickle(f"{path}/y_scaler")
        svd.n_rf = from_pickle(f"{path}/n_rf")
        svd.history = from_pickle(f"{path}/history")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        self.model.save(f"{path}/model")
        to_pickle(self.X_scaler, f"{path}/X_scaler")
        to_pickle(self.y_scaler, f"{path}/y_scaler")
        to_pickle(self.n_rf, f"{path}/n_rf")
        to_pickle(self.history, f"{path}/history")


class ReverseRFDisturbedANN(ReverseRF):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.X2_scaler = MinMaxScaler()
        self.y_scaler = StandardScaler()
        
        self.model = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(self.n_rf, activation=None)]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    def fit(self, rf_disturbed, formfactors, rf, epochs=1000, verbose=1):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        y = self._preprocess_rf(rf)
        
        y_scaled = self.y_scaler.fit_transform(y)
        X1_scaled = self.y_scaler.transform(X1)
        X2_scaled = self.X2_scaler.fit_transform(X2)
        
        X_scaled = np.concatenate([X1_scaled,X2_scaled], axis=1)
        
        history = self.model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=64, verbose=verbose)
        
        self.history = history.history
            
    def predict(self, rf_disturbed, formfactors):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        
        X1_scaled = self.y_scaler.transform(X1)
        X2_scaled = self.X2_scaler.transform(X2)
        
        X_scaled = np.concatenate([X1_scaled,X2_scaled], axis=1)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        return y
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model = keras.models.load_model(f"{path}/model")
        svd.X2_scaler = from_pickle(f"{path}/X2_scaler")
        svd.y_scaler = from_pickle(f"{path}/y_scaler")
        svd.n_rf = from_pickle(f"{path}/n_rf")
        svd.history = from_pickle(f"{path}/history")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        self.model.save(f"{path}/model")
        to_pickle(self.X2_scaler, f"{path}/X2_scaler")
        to_pickle(self.y_scaler, f"{path}/y_scaler")
        to_pickle(self.n_rf, f"{path}/n_rf")
        to_pickle(self.history, f"{path}/history")


class ReverseRFDeltaDisturbedANN(ReverseRF):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.X1_scaler = StandardScaler()
        self.X2_scaler = MinMaxScaler()
        self.y_scaler = StandardScaler()
        
        self.model = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(self.n_rf, activation=None)]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    def fit(self, rf_disturbed, formfactors, rf, epochs=1000, verbose=1):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        y = X1 - self._preprocess_rf(rf)
        
        X1_scaled = self.X1_scaler.fit_transform(X1)
        X2_scaled = self.X2_scaler.fit_transform(X2)
        y_scaled = self.y_scaler.fit_transform(y)
        
        X_scaled = np.concatenate([X1_scaled,X2_scaled], axis=1)
        
        history = self.model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=64, verbose=verbose)
        
        self.history = history.history
            
    def predict(self, rf_disturbed, formfactors):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        
        X1_scaled = self.X1_scaler.transform(X1)
        X2_scaled = self.X2_scaler.transform(X2)
        
        X_scaled = np.concatenate([X1_scaled,X2_scaled], axis=1)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        correct_rf = X1 - y

        return correct_rf
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model = keras.models.load_model(f"{path}/model")
        svd.X1_scaler = from_pickle(f"{path}/X1_scaler")
        svd.X2_scaler = from_pickle(f"{path}/X2_scaler")
        svd.y_scaler = from_pickle(f"{path}/y_scaler")
        svd.n_rf = from_pickle(f"{path}/n_rf")
        svd.history = from_pickle(f"{path}/history")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        self.model.save(f"{path}/model")
        to_pickle(self.X1_scaler, f"{path}/X1_scaler")
        to_pickle(self.X2_scaler, f"{path}/X2_scaler")
        to_pickle(self.y_scaler, f"{path}/y_scaler")
        to_pickle(self.n_rf, f"{path}/n_rf")
        to_pickle(self.history, f"{path}/history")

    
class ReverseRFKNN(ReverseRF):
    
    def __init__(self, n_neighbors=5, weights="uniform", **kwargs):
        super().__init__(**kwargs)
        
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    
    def fit(self, formfactors, rf, epochs=1000, verbose=1):
        X = self._preprocess_formfactors(formfactors)
        y = self._preprocess_rf(rf)
        
        self.history = self.model.fit(X, y)
            
    def predict(self, formfactors):
        X = self._preprocess_formfactors(formfactors)
        y = self.model.predict(X)
        return y
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model = from_pickle(f"{path}/model")
        svd.n_rf = from_pickle(f"{path}/n_rf")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        to_pickle(self.model, f"{path}/model")
        to_pickle(self.n_rf, f"{path}/n_rf")


class ReverseRFDisturbedKNN(ReverseRFKNN):
    
    def fit(self, rf_disturbed, formfactors, rf):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        y = self._preprocess_rf(rf)
        
        self.history = self.model.fit(X, y)
    
    def predict(self, rf_disturbed, formfactors):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        y = self.model.predict(X)
        return y


class ReverseRFDeltaDisturbedKNN(ReverseRFKNN):
    
    def fit(self, rf_disturbed, formfactors, rf):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        y = X1 - self._preprocess_rf(rf)
        
        self.history = self.model.fit(X, y)
    
    def predict(self, rf_disturbed, formfactors):
        X1 = self._preprocess_rf(rf_disturbed)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)

        y = self.model.predict(X)

        correct_rf = X1 - y

        return correct_rf


class LockmANN(AdaptiveANNTHz):

    def predict(self, formfactors):
        clean = []
        for frequency, formfactor, formfactor_noise, detlim, _ in formfactors:
            clean.append(cleanup_formfactor(frequency, formfactor, formfactor_noise, detlim, channels_to_remove=[])[:2])
        ann_currents = super().predict(clean)

        nils_currents = []
        for frequencies, formfactors, formfactor_noise, detlim, charge in formfactors:
            reconstructed = master_recon(frequencies, formfactors, formfactor_noise,
                                         detlim, charge=charge, method="KKstart",
                                         channels_to_remove=[], show_plots=False)
            recon_time, recon_current = reconstructed[:2]                                       
            centered = self.center_current((recon_time*3e8, recon_current))
            nils_currents.append(centered)
        
        finals = []
        for ann, nils in zip(ann_currents, nils_currents):
            flipped = (nils[0], np.flip(nils[1]))

            mae = self.compute_mae(ann, nils)
            mae_flipped = self.compute_mae(ann, flipped)

            finals.append(flipped if mae_flipped < mae else nils)
        
        return np.array(finals)

    def compute_mae(self, a, b):
        n = 0
        left = min(a[0].min(), b[0].min())
        right = max(a[0].max(), b[0].max())
        
        new_s = np.linspace(left, right, 100)
        ti = np.interp(new_s, a[0], a[1], left=0, right=0)
        pi = np.interp(new_s, b[0], b[1], left=0, right=0)

        mae = np.abs(ti - pi).mean()
        return mae
    
    def center_current(self, current):
        left, right = find_edges(current[0], current[1])
        resampled = resample(current[0], current[1], left, right, self.n_samples)
        centered = center_on_zero(resampled[0], resampled[1], left, right)
        return centered


class Peak:
    
    def _preprocess_rf(self, rf):
        X = np.stack(rf)
        return X
        
    def _preprocess_formfactors(self, formfactors):
        X = np.stack([formfactor for _, formfactor in formfactors])
        return X
    
    def _preprocess_peaks(self, peaks):
        return np.array(peaks).reshape((-1, 1))
    
    
class PeakANN(Peak):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        self.model = keras.Sequential([
            layers.Dense(200, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(50, activation="relu"),
            layers.Dense(1, activation="relu")]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model = keras.models.load_model(f"{path}/model")
        svd.X_scaler = from_pickle(f"{path}/X_scaler")
        svd.y_scaler = from_pickle(f"{path}/y_scaler")
        svd.history = from_pickle(f"{path}/history")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        self.model.save(f"{path}/model")
        to_pickle(self.X_scaler, f"{path}/X_scaler")
        to_pickle(self.y_scaler, f"{path}/y_scaler")
        to_pickle(self.history, f"{path}/history")


class PeakANNRF(PeakANN):
    
    def fit(self, rf, peaks, epochs=1000, verbose=1):
        X = self._preprocess_rf(rf)
        y = self._preprocess_peaks(peaks)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        history = self.model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=64, verbose=verbose)

        self.history = history.history
    
    def predict(self, rf):
        X = self._preprocess_rf(rf)
        X_scaled = self.X_scaler.transform(X)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        return y


class PeakANNTHz(PeakANN):
    
    def fit(self, formfactors, peaks, epochs=1000, verbose=1):
        X = self._preprocess_formfactors(formfactors)
        y = self._preprocess_peaks(peaks)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        history = self.model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=64, verbose=verbose)

        self.history = history.history
            
    def predict(self, formfactors):
        X = self._preprocess_formfactors(formfactors)
        X_scaled = self.X_scaler.transform(X)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        return y
    

class PeakANNRFTHz(PeakANN):
    
    def fit(self, rf, formfactors, peaks, epochs=1000, verbose=1):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        y = self._preprocess_peaks(peaks)
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        history = self.model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=64, verbose=verbose)

        self.history = history.history
            
    def predict(self, rf, formfactors):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        X_scaled = self.X_scaler.transform(X)
        
        y_scaled = self.model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        return y


class PeakKNN(Peak):
    
    def __init__(self, n_neighbors=2, weights="distance", **kwargs):
        super().__init__(**kwargs)
        
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    
    @classmethod
    def load(cls, path):
        svd = cls()
        svd.model = from_pickle(f"{path}/model")
        
        return svd
    
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        to_pickle(self.model, f"{path}/model")
    

class PeakKNNRF(PeakKNN):
    
    def fit(self, rf, peaks):
        X = self._preprocess_rf(rf)
        y = self._preprocess_peaks(peaks)
        
        self.model.fit(X, y)
    
    def predict(self, rf):
        X = self._preprocess_rf(rf)
        y = self.model.predict(X)
        return y


class PeakKNNTHz(PeakKNN):
    
    def fit(self, formfactors, peaks):
        X = self._preprocess_formfactors(formfactors)
        y = self._preprocess_peaks(peaks)
        
        self.model.fit(X, y)
    
    def predict(self, formfactors):
        X = self._preprocess_formfactors(formfactors)
        y = self.model.predict(X)
        return y


class PeakKNNRFTHz(PeakKNN):
    
    def fit(self, rf, formfactors, peaks):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        y = self._preprocess_peaks(peaks)
        
        self.model.fit(X, y)
    
    def predict(self, rf, formfactors):
        X1 = self._preprocess_rf(rf)
        X2 = self._preprocess_formfactors(formfactors)
        X = np.concatenate([X1,X2], axis=1)
        
        y = self.model.predict(X)
        
        return y
