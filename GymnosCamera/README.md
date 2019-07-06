# GymnosCamera

Library for camera and predictions

## Development

To install this library with local changes:

```bash
pip install -e .
```
or (if you are in a different directory)
```bash
pip install -e path/to/GymnosCamera
```

To uninstall this library, simply run `pip uninstall gymnoscamera`

## Building

To build this library, ensure you are in a virtualenv, and install the requirements from the main
directory.

Run the following command from this directory:

```bash
python3 setup.py sdist
```

## Publishing

To publish this library, ensure you are in a virtualenv, and install 'devpi-client'

```bash
pip install devpi-client
```

Login as user 'gymnos' (see google drive 'Shared Credentials' for password)

```bash
> devpi login gymnos
password for user gymnos:
logged in 'gymnos', credentials valid for 10.00 hours
```

Choose which repository to upload to:

1) Stable (cannot overwrite uploaded versions!)
    ```bash
    use gymnos/stable
    ```
2) Staging (can overwrite uploaded versions)
    ```bash
    use gymnos/staging
    ```

Ensure you are in the GymnosCamera directory and upload using devpi:

```bash
devpi upload
```