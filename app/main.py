from fastapi import FastAPI, HTTPException, status
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
    #if text empty, return http exception
    if len(text) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Activity is empty")
    
    model = load_model('app/model/lstm_model.h5')
    preprocessed_text = data_preprocessing(text)
    padded_sequences = tokenize(preprocessed_text)
    result = model.predict(padded_sequences)
    weight = 0
    denominator = 0
    for i in range(len(result)):
        weight += (i+1) * result[i]
        denominator += (i+1)
    result = weight / denominator

    result = np.mean(result, axis=0)
    
    return result.tolist()



