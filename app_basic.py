#streamlit run app_basic.py

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from get_model import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
resnet = get_model("resnet50")
resnet.load_state_dict(torch.load("checkpoints/resnet50_best.pth", map_location=device))
resnet.to(device).eval()

dense = get_model("densenet121")
dense.load_state_dict(torch.load("checkpoints/densenet121_best.pth", map_location=device))
dense.to(device).eval()

# Common transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("X-Ray Classification")

model_choice = st.radio("Select Model", ["ResNet50", "DenseNet121"])
uploaded = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    inp = transform(img).unsqueeze(0).to(device)

    if st.button("Predict"):
        model = resnet if model_choice == "ResNet50" else dense
        with torch.no_grad():
            output = model(inp)
            _, pred = output.max(1)
        
        st.success(f"Prediction: Class {pred.item()}")