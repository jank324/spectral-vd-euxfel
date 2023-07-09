import pickle

import numpy as np
import torch

from legacy import SupervisedCurrentProfileInference

# Load models and scalers used by the reconstruction function
model = SupervisedCurrentProfileInference("epoch=76-step=29876.ckpt").eval()
with open("train_scalers_current.pkl", "rb") as f:
    scalers = pickle.load(f)


def reconstruct_current_profile(
    rf_settings: np.ndarray, formfactor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs the current profile from the RF settings and formfactor using a neural
    network model.

    :param rf_settings: RF settings used for the reconstruction. Must be NumPy array of
        the form [chirp, curv, skew, chirpL1, chirpL2].
    :param formfactor: Formfactor used for the reconstruction. Must be NumPy array of
        shape (240,).
    :return: Tuple (s, I) where s in given in meters and I in Amperes. Both are NumPy
        arrays of shape (300,).
    """
    assert rf_settings.shape == (5,)
    assert formfactor.shape == (240,)

    # Normalize and reshape inputs, then convert to PyTorch tensors
    x_rf = scalers["rf"].transform(rf_settings.reshape(1, -1))
    x_formfactor = scalers["formfactor"].transform(formfactor.reshape(1, -1))

    x_rf = torch.tensor(x_rf, dtype=torch.float32)
    x_formfactor = torch.tensor(x_formfactor, dtype=torch.float32)

    # Run the model
    y_hat_current_profile, y_hat_bunch_length = model(x_rf, x_formfactor)

    # Rescale the outputs and convert back to NumPy arrays
    predicted_current_profile = scalers["current"].inverse_transform(
        y_hat_current_profile.detach().numpy()
    )
    predicted_bunch_length = scalers["bunch_length"].inverse_transform(
        y_hat_bunch_length.detach().numpy()
    )

    # Build the s sample points
    predicted_ss = np.linspace(
        -predicted_bunch_length[0][0] / 2, predicted_bunch_length[0][0] / 2, num=300
    )

    return predicted_ss, predicted_current_profile[0]
