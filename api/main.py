from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib

# ----- CONFIG -----
INPUT_DIM = 16  # ← replace with your actual input feature count

# ----- MODEL -----
class AirbnbRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# ----- LOAD MODEL + SCALERS -----
model = AirbnbRegressor(INPUT_DIM)
model.load_state_dict(torch.load("../model/airbnb_model.pt", map_location=torch.device('cpu')))
model.eval()

input_scaler = joblib.load("../data/scaler.pkl")
price_scaler = joblib.load("../data/price_scaler.pkl")

# ----- FASTAPI APP -----
app = FastAPI()

# Input schema
class Listing(BaseModel):
    features = [
        40.68514,  # latitude
        -73.95976,  # longitude
        1.0,  # Example: minimum_nights
        270.0,  # number_of_reviews
        4.64,  # reviews_per_month
        294.0,  # availability_365
        551.0,  # encoding
    ]  # Must match order of features used during training

    second_features = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

@app.post("/predict")
def predict_price(listing: Listing):
    x = np.array(listing.features).reshape(1, -1)
    x_scaled = input_scaler.transform(x)
    x_added = np.array(listing.second_features).reshape(1, -1)
    final_input = np.concatenate((x_scaled, x_added), axis=1)
    x_tensor = torch.tensor(final_input, dtype=torch.float32)

    with torch.no_grad():
        scaled_output = model(x_tensor).item()
        real_price = price_scaler.inverse_transform([[scaled_output]])[0][0]

    return {"predicted_price": round(real_price, 2)}
