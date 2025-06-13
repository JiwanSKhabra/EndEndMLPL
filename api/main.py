from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import numpy as np
import joblib
import uvicorn
import os

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

model_path = os.path.join(BASE_DIR, "../model/airbnb_model.pt")
scaler_path = os.path.join(BASE_DIR, "../data/scaler.pkl")
price_scaler_path = os.path.join(BASE_DIR, "../data/price_scaler.pkl")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Load model and scalers
INPUT_DIM = 16
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


model = AirbnbRegressor(INPUT_DIM)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

input_scaler = joblib.load(scaler_path)
price_scaler = joblib.load(price_scaler_path)

# GET: Show form
@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# POST: Handle form submission
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_from_form(
    request: Request,
    latitude: float = Form(...),
    longitude: float = Form(...),
    minimum_nights: float = Form(...),
    number_of_reviews: float = Form(...),
    reviews_per_month: float = Form(...),
    availability_365: float = Form(...),
    neighbourhood_encoded: float = Form(...),
    is_available: int = Form(...),
    room_type: str = Form(...),
    neighbourhood_group: str = Form(...)
):
    # One-hot encoding
    room_type_vec = {
        "Entire home/apt": [1, 0, 0],
        "Private room": [0, 1, 0],
        "Shared room": [0, 0, 1]
    }[room_type]

    neighbourhood_group_vec = {
        "Bronx": [1, 0, 0, 0, 0],
        "Brooklyn": [0, 1, 0, 0, 0],
        "Manhattan": [0, 0, 1, 0, 0],
        "Queens": [0, 0, 0, 1, 0],
        "Staten Island": [0, 0, 0, 0, 1]
    }[neighbourhood_group]

    base_features = [
        latitude, longitude, minimum_nights, number_of_reviews,
        reviews_per_month, availability_365, neighbourhood_encoded
    ]
    second_features = [is_available] + room_type_vec + neighbourhood_group_vec

    x = np.array(base_features).reshape(1, -1)
    x_scaled = input_scaler.transform(x)
    x_cat = np.array(second_features).reshape(1, -1)
    final_input = np.concatenate((x_scaled, x_cat), axis=1)

    x_tensor = torch.tensor(final_input, dtype=torch.float32)
    with torch.no_grad():
        scaled_output = model(x_tensor).item()
        real_price = price_scaler.inverse_transform([[scaled_output]])[0][0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "predicted_price": round(real_price, 2)
    })
