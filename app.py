# app.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
from PIL import Image
import io

# If your model is PyTorch
import torch
from torchvision import transforms

app = FastAPI()

# ---- Load Model ----
model = torch.load("swimming_model.pth", map_location="cpu")
model.eval()

# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # match preprocessing in your notebook
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Run model
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(1).item()

    return {"prediction": int(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
