from fastapi import FastAPI
import numpy as np
from typing import List
from tensorflow.keras.models import load_model
from app.services.preprocess import data_preprocessing
from app.services.tokenizer import tokenize

app = FastAPI()

#cors
from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(text: List[str]):
    model = load_model('app/model/bidirectional_lstm.h5')
    preprocessed_text = data_preprocessing(text)
    padded_sequences = tokenize(preprocessed_text)
    result = model.predict(padded_sequences)
    
    result = np.mean(result, axis=0)
    
    return result.tolist()[0]



