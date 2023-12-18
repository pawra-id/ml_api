# ML API
This is repository for machine learning backend. Using FastAPI and Tensorflow.

## Requiremenets
- Python >= 3.10

## Install
- Clone the repo
- run ```pip install --no-cache-dir -r requirements.txt``` in terminal
- run ```pip install tensorflow``` in terminal
- Finally, run ```uvicorn app.main:app --reload --port 8000```
- Access the app at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Import New Model
- Get the h5 file from retraining the code from [ML Repository](https://github.com/pawra-id/ml_repo)
- Copy the h5 file into the ```app/model/```
- Make sure to rename it as ```bidirectional_lstm.h5```
