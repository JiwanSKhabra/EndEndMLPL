import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv("FinalCleaned.csv")

X = torch.tensor(df.drop(columns=['price_scaled']).values, dtype=torch.float32)
y = torch.tensor(df['price_scaled'].values, dtype=torch.float32).view(-1, 1)

# Move to device
X, y = X.to(device), y.to(device)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X.cpu(), y.cpu(), test_size=0.2, random_state=42)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# DataLoaders
batch_size = 64
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# Neural network with Dropout
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

# Initialize model, loss, optimizer, scheduler
model = AirbnbRegressor(X.shape[1]).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_train = model(X_train)
        preds_test = model(X_test)
        rmse_train = mean_squared_error(y_train.cpu(), preds_train.cpu())**0.5
        rmse_test = mean_squared_error(y_test.cpu(), preds_test.cpu())**0.5

    print(f"Epoch {epoch+1}/{epochs} | Train RMSE: {rmse_train:.4f} | Test RMSE: {rmse_test:.4f}")


torch.save(model.state_dict(), "airbnb_model.pt")

print("Model and input scaler saved.")
