import pickle
import random

import numpy as np
import pydoocs
from scipy.constants import speed_of_light
from tensorflow import keras

from nils.reconstruction_module import cleanup_formfactor, master_recon
from nils.simulate_spectrometer_signal import get_crisp_signal


class SpectralVD:

    is_simulation_mode = True   # Set true to run from fake fomrfactors rather than PyDoocs

    def __init__(self):
        self.model = keras.models.load_model("model")
        with open("scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)
        
        if self.is_simulation_mode:
            self.load_formfactors()

    def load_formfactors(self):
        with open("../research/ocelot80k.pkl", "rb") as file:
            data = pickle.load(file)
        
        currents = [(sample["s"][:1000], sample["I"][:1000]) for sample in data]
        filtered = [(s, current) for s, current in currents if current.max() > 1000]

        samples = random.choices(filtered, k=10)
        self.crisp_both = [get_crisp_signal(s, current, n_shots=10, which_set="both") for s, current in samples]
    
    def read_crisp(self):
        if self.is_simulation_mode:
            self.crisp_reading = random.choice(self.crisp_both)
        else:
            self.crisp_reading = (      # TODO: Add actual DOOCS adresses
                pydoocs.read("frequency_foobar")["data"],
                np.sqrt(pydoocs.read("formfactor_foobar")["data"]),
                pydoocs.read("formfactor_noise_foobat")["data"],
                pydoocs.read("detlim_foobar")["data"]
            )
    
    def ann_reconstruction(self):
        frequency, formfactor, formfactor_noise, detlim = self.crisp_reading

        _, clean_formfactor, _ = cleanup_formfactor(frequency, formfactor, formfactor_noise, detlim, channels_to_remove=[])

        X = clean_formfactor.reshape([1,-1])
        X_scaled = self.scalers["X"].transform(X)
        y_scaled = self.model.predict(X_scaled)
        y = y_scaled / self.scalers["y"]
        current = y.squeeze()

        limit = 0.00020095917745111108
        s = np.linspace(-limit, limit, 100)

        return s, current

    def nils_reconstruction(self):
        frequency, formfactor, formfactor_noise, detlim = self.crisp_reading
        charge = 250e-12

        t, current, _ = master_recon(frequency, formfactor, formfactor_noise, detlim, charge,
                                     method="KKstart", channels_to_remove=[], show_plots=False)

        s = t * speed_of_light

        return s, current
