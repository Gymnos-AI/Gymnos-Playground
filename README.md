# Gymnos Repo
- This is a repository to encapsulate all of the Gymnos products.

### Modularity Notes
The GymnosCamera (gymnoscamera) sub-module should be split into a separate repository
as it acts like a library for GymnosServer and a full-fledged system for USB web cam
or Pi camera capable devices.

## Setting up a Virtual Environment

### Windows Setup:
1) Create the virtual environment (first time only):
    ```
    py -3.7 -m venv venv
    ```

2) Activate it (for every new terminal): 
    ```
    .\venv\Scripts\activate
    ```

### Linux Setup:
1) Create the virtual environment (first time only):
    ```bash
    python3 -m venv venv
    ```

2) Activate it (for every new terminal):
    ```bash
    source ./venv/bin/activate
    ```

### Deactivating the virtual-env
1) Run the deactivate command:
    ```
    deactivate
    ```

### Installing dependencies
1) Install requirements file:
    ```
    pip install -r requirements.txt
    ```

## Run the systems

To run the camera, run either `run-camera.bat` or `run-camera.sh` according to your environment.
Make sure to have a virtual environment activated!