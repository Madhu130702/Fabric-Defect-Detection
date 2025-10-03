import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import tempfile
import os
import gdown

# ======================
# 0. Config
# ======================
st.set_page_config(page_title="Fabric Defect Detection App", layout="centered")

# Google Drive file IDs (from your links)
CNN_ID  = "1z8fnAar1xc_aDdXg3UgD-EbbGKjKo6kT"
YOLO_ID = "1r8POhU1ItnqlJVrLPrMVqkQsk4z8fFhA"

CNN_WEIGHTS  = "cnn_model_new.pth"
YOLO_WEIGHTS = "best.pt"

# ======================
# Utilities: download once, reuse thereafter
# ======================
def download_if_missing(file_id: str, out_path: str) -> str:
    """Download a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(out_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Downloading {out_path} ..."):
            gdown.download(url, out_path, quiet=False)
        if not os.path.exists(out_path):
            st.error(f"Failed to download {out_path}. Check sharing permissions (Anyone with link: Viewer).")
            st.stop()
    return out_path

# ======================
# 1. Define CNN Model
# ======================
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),  # 256 -> 128 -> 64 -> 32
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)

# ======================
# 2. Preprocessing
# ======================
def preprocess_image(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    return img_tensor

# ======================
# 3. Load CNN Model (cached)
# ======================
@st.cache_resource(show_spinner=False)
def load_cnn_model() -> nn.Module:
    weights_path = download_if_missing(CNN_ID, CNN_WEIGHTS)
    model = BetterCNN()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ======================
# 4. Load YOLO Model (cached)
# ======================
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    weights_path = download_if_missing(YOLO_ID, YOLO_WEIGHTS)
    return YOLO(weights_path)

# ======================
# 5. UI
# ======================
st.title("🧵 Fabric Defect Classifier + Localizer")
st.markdown("Upload a fabric image to **classify** and, if defective, **localize** defects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("⏳ Classifying...")
    cnn_model = load_cnn_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = cnn_model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    clean_conf = float(probs[0].item())
    defect_conf = float(probs[1].item())
    predicted_class = int(torch.argmax(probs).item())
    label_map = {1: "⚠️ Defective Fabric", 0: "✅ Clean Fabric"}

    st.write(f"🧪 Confidence — Clean: {clean_conf:.4f} | Defective: {defect_conf:.4f}")
    if predicted_class == 1:
        st.warning(f"Prediction: **{label_map[predicted_class]}**")
    else:
        st.success(f"Prediction: **{label_map[predicted_class]}**")

    # If defective → run YOLO
    if predicted_class == 1:
        st.write("🔍 Running YOLOv8 to locate defects...")
        yolo_model = load_yolo_model()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            results = yolo_model.predict(tmp_file.name, conf=0.25, verbose=False)

        # Show YOLO result image(s)
        for r in results:
            # Save a plotted image to a temp path and display
            out_path = os.path.join(tempfile.gettempdir(), "yolo_result.jpg")
            r.save(filename=out_path)  # Ultralytics v8 supports save(filename=...)
            st.image(out_path, caption="Detected Defects", use_column_width=True)

else:
    st.info("Upload a JPG/PNG image to start.")
