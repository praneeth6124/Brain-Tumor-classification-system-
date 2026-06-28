import torch
from PIL import Image
from torchvision import transforms
from backend.model import create_model
from backend.class_names import CLASS_NAMES

# Load model once
model = create_model()

model.load_state_dict(
    torch.load("models/best_model.pth", map_location="cpu")
)

model.eval()

# Same image size used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    confidence, pred = torch.max(probs, dim=1)

    confidence_score = confidence.item() * 100

    if confidence_score >= 90:
        status = "Prediction is Reliable"

    elif confidence_score >= 70:
        status = "acceptable prediction confidence"

    else:
        status = "Uncertain Prediction - Further Evaluation Recommended.."

    all_probs = probs[0].tolist()

    probabilities = {
        CLASS_NAMES[i]: round(all_probs[i] * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": round(confidence_score, 2),
        "status": status,
        "probabilities": probabilities
    }

if __name__ == "__main__":
    result = predict_image("test_images/sample.jpg")

    print(result)