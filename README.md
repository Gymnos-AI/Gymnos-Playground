# Gymnos Repo
- This is a repository to encapsulate all of the Gymnos products.
- GymnosCamera has its own repo now: https://github.com/JamesPeralta/GymnosCamera

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
2) Install Tensorflow. Depending on your computer choose **a)** or **b)**

    a) If your computer does not have a NVIDIA GPU
    ```
    tensorflow==1.14.0
    ```
    b) If your computer has a NVIDIA GPU
    ```
    tensorflow-gpu==1.14.0
    ```
    
## Run the systems

To run the camera, run either `run-camera.bat` or `run-camera.sh` according to your environment.
Make sure to have a virtual environment activated and the gymnoscamera package installed!
