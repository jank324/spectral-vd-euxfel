# Spectral Virtual Diagnostics and European XFEL

Virtual Diagnostics for inferring current profiles of bunches at European XFEL

## Organisation of the Repository

The app to run in BKR and everything needed to actually run in in BKR and on the accelerator should be in the `bkr` directory. The `research` directory contains everything else used to develop the app. The latter may be messy.

In the `bkr` directory you will find a file `spectralvd.py`. This contains all the logic to talk to the accelerator and run the current inference. The file `app.py` constructs the PyQt5 UI and connects it to the logic in `spectralvd.py`. The `app.py` is also the entry point to the application. The `model` directory contains a trained model for current inference. The model requires scaling of its inputs and outputs. The required scalers are saved in `scalers.pkl`.

## Run in BKR

To run the application in BKR, you need to change into the `bkr` directory and run `app.py` as follows:

```bash
cd bkr
python3 app.py
```

As it stands, the app runs in simulation mode, meaning it is fed random simulated CRISP formfactors. In order to actually read data from DOOCS, find `spectralvd.py` and change the variable `is_simulation_mode` in line 16 to `False`.

You may be missing Python packages. A `requirements.txt` can be found in the `bkr` directory.
