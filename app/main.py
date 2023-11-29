from fastapi import FastAPI
from tensorflow.keras.models import load_model
from app.services.preprocess import data_preprocessing
from app.services.tokenizer import tokenize

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(text: str):
    model = load_model('app/model/bidirectional_lstm.h5')
    preprocessed_text = data_preprocessing(text)
    padded_sequences = tokenize([preprocessed_text])
    result = model.predict(padded_sequences)
    
    return result.tolist()[0][0]



