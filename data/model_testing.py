import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# Load scalers
input_scaler = joblib.load("scaler.pkl")
price_scaler = joblib.load("price_scaler.pkl")

# Load trained model
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

# Load model and weights
input_dim = 16 # Replace with your actual input feature count
model = AirbnbRegressor(input_dim=input_dim)
model.load_state_dict(torch.load("airbnb_model.pt", map_location=torch.device("cpu")))
model.eval()

# 🧪 Replace this with your real input (scaled manually later)
sample_input = [
    40.68514,  # latitude
    -73.95976, # longitude
    1.0,  # Example: minimum_nights
    270,  # number_of_reviews
    4.64,  # reviews_per_month
    294,  # availability_365
    551, # encoding
]

print('hello')
# Convert to 2D array and scale
sample_input = np.array(sample_input).reshape(1, -1)
sample_input_scaled = input_scaler.transform(sample_input)
added_input = [1, 1, 0, 0, 0, 1, 0, 0, 0]
added_input = np.array(added_input).reshape(1, -1)

print('why')
final_tesor = np.concatenate((sample_input_scaled, added_input), axis=1)
print('plz')

print(final_tesor)


# Convert to tensor
x_tensor = torch.tensor(final_tesor, dtype=torch.float32)

# Predict
with torch.no_grad():
    scaled_price = model(x_tensor).item()
    predicted_price = price_scaler.inverse_transform([[scaled_price]])[0][0]

print(f"Predicted price: ${predicted_price:.2f}")
