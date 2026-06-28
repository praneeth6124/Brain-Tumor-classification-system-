import torch
from model import create_model

model = create_model()

model.load_state_dict(
    torch.load("models/best_model.pth", map_location="cpu")
)

model.eval()

print("Model loaded successfully!")