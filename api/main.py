from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Big Small Big World"}

@app.get("/predict")
def predict():
    return {"prediction": "Model not loaded yet"}
