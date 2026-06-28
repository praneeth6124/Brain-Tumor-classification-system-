import torch

model_path = "models/best_model.pth"

state_dict = torch.load(model_path, map_location="cpu")

print("Model loaded successfully!")
print(type(state_dict))
