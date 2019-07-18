@echo off

set PYTHONUNBUFFERED=1
set PYTHONPATH=.
python GymnosServer\Flask\PredictorsAPI.py --model yolo.h5