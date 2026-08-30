# 🧵 Fabric Defect Detection

A Streamlit web app that classifies fabric images as **clean** or **defective**, and — when a defect is detected — localizes it on the image using a YOLOv8 model. Built as a two-stage computer vision pipeline: a custom CNN handles the classification step, and YOLOv8 handles defect localization.

## How it works

1. **Upload** a fabric image (JPG/PNG) through the Streamlit interface.
2. **Classify** — a custom CNN (`BetterCNN`) predicts whether the fabric is clean or defective, along with a confidence score for each class.
3. **Localize** — if the image is classified as defective, a YOLOv8 model runs on the same image and draws bounding boxes around the detected defect regions.
4. **Display** — the app shows the classification result, confidence scores, and (if applicable) the annotated defect image.

## Model architecture

**Classifier — `BetterCNN`**
A convolutional neural network built in PyTorch with three convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool, with 16 → 32 → 64 channels), followed by a fully connected head that outputs a 2-class prediction (clean vs. defective). Input images are resized to 256×256 before inference.

**Localizer — YOLOv8**
A YOLOv8 model (via the `ultralytics` package) trained to detect and box defect regions on fabric images. It only runs when the CNN flags an image as defective, keeping the common case (clean fabric) fast.

Both sets of model weights are hosted externally (Google Drive) and are downloaded automatically on first run via `gdown`, then cached locally so subsequent runs don't re-download them.

## Tech stack

- **App framework:** Streamlit
- **Deep learning:** PyTorch, torchvision
- **Object detection:** Ultralytics YOLOv8
- **Image handling:** Pillow
- **Weight retrieval:** gdown (Google Drive downloads)

## Project structure

```
Fabric-Defect-Detection/
├── app.py              # Streamlit app: UI, model loading, inference pipeline
├── requirements.txt     # Python dependencies
├── packages.txt          # System-level dependency (libgl1, needed by OpenCV/YOLO)
└── runtime.txt           # Python runtime version for deployment
```

## Getting started

### Prerequisites
- Python 3.9+ (see `runtime.txt` for the exact version used)
- pip

### Installation

```bash
git clone https://github.com/Madhu130702/Fabric-Defect-Detection.git
cd Fabric-Defect-Detection
pip install -r requirements.txt
```

> Note: `requirements.txt` pins the CPU build of PyTorch via `--extra-index-url https://download.pytorch.org/whl/cpu`. If you have a CUDA-capable GPU and want GPU acceleration, install a matching `torch`/`torchvision` build separately before running `pip install -r requirements.txt`.

If deploying to a Linux environment without a display server (e.g. Streamlit Community Cloud), make sure the system package in `packages.txt` (`libgl1`) is installed — it's required by OpenCV/YOLO for image processing.

### Running the app

```bash
streamlit run app.py
```

The app will open in your browser. On first run, it will download the CNN and YOLOv8 weights from Google Drive (a few seconds to a couple of minutes depending on connection speed) and cache them locally as `cnn_model_new.pth` and `best.pt`.

### Usage

1. Open the app in your browser.
2. Upload a `.jpg`, `.jpeg`, or `.png` image of a fabric sample.
3. The app classifies it as clean or defective and shows confidence scores.
4. If defective, the app automatically runs YOLOv8 and displays the image with bounding boxes around the detected defects.

## Limitations

- Classification is binary (clean vs. defective) and doesn't distinguish between defect types at the CNN stage — defect typing is left to the YOLOv8 detection labels.
- Model weights are fetched from Google Drive at runtime rather than bundled in the repo, so the app requires an internet connection on first launch.
- Performance depends on how closely uploaded images resemble the training data (lighting, fabric type, image resolution).

## Future improvements

- Bundle or version model weights with the repo (or via Git LFS / releases) to remove the runtime download dependency.
- Add defect-type classification alongside localization.
- Add a batch-upload mode for processing multiple fabric images at once.
