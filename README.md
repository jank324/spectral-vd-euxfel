# Spectral Virtual Diagnostics and European XFEL

Virtual Diagnostics for inferring current profiles of bunches at European XFEL

## How to Run the App in BKR

There are two ways to run the app in BKR. You can either run it remotely via one of the `xfeluser` servers or locally on either a macOS or Linux console. The first option is probably quicker to get up and running, whereas the latter probably runs more smoothly.

### Via `xfeluser` Server

To run remotely simply connect to one of the `xfeluser` servers

```bash
ssh username@xfeluser2
```

activate my Anaconda environment

```bash
conda activate /home/kaiserja/.conda/envs/spectral-vd
```

and start the app

```bash
python /home/kaiserja/spectral-vd-euxfel/app.py
```

### Locally on Console

If you are running on a local Linux console you may be able to activate the environment and run the app as you would on the `xfeluser` servers (see above). On a macOS console you will have to create a suitable Anaconda environment. You can create an Anaconda environment on Linux as well, if you wisch to do so.

In order to create a new Anaconda environment that is configured correctly, run the following commands in order:

```bash
git clone https://github.com/jank324/spectral-vd-euxfel.git
cd spectral-vd-euxfel
conda create --name spectral-vd python=3.9
conda activate spectral-vd
pip install requirements.txt
```

Then run

```bash
python app.py
```

to run the app.
