from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


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
    