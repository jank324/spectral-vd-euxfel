from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from .utils import current2formfactor


def load_data(path: Union[Path, str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the current reconstruction data set given a `pandas.DataFrame` saved as a
    `.pkl` file at `path`.
    """
    path = Path(path) if isinstance(path, str) else path

    df = pd.read_pickle(path)

    rfs_dataset = df[["chirp", "chirpL1", "chirpL2", "curv", "skew"]].values

    ss_dataset = np.stack(
        [np.linspace(0, 300 * df.loc[i, "slice_width"], num=300) for i in df.index]
    )

    currents_dataset = np.stack(df["slice_I"].values)

    stacked_currents_dataset = np.stack([ss_dataset, currents_dataset], axis=1)

    formfactors_dataset = np.array(
        [
            current2formfactor(ss, currents, grating="both", n_shots=1, clean=False)
            for ss, currents in zip(ss_dataset, currents_dataset)
        ]
    )
    formfactors_dataset_reshaped = formfactors_dataset.reshape(-1, 480)

    X = np.concatenate([rfs_dataset, formfactors_dataset_reshaped], axis=1)
    y = stacked_currents_dataset.reshape(-1, 600)

    return X, y


def main() -> None:
    # Train
    X_train, y_train = load_data("data/zihan/train.pkl")
    knn = KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)

    # Test
    X_train, y_train = load_data("data/zihan/test.pkl")
    y_pred = knn.predict(X_train[:1])

    print(f"{y_pred = }")


if __name__ == "__main__":
    main()
