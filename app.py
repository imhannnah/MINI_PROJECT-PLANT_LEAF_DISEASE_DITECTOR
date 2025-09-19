
import os, io, json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
import cv2

# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
UPLOADS = BASE_DIR / "static" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
ALLOWED = {"png","jpg","jpeg","bmp","webp"}

# ---------- Helpers ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED

def load_idx_to_class(p: Path):
    with open(p, "r") as f:
        class_to_idx = json.load(f)
    return {v: k for k, v in class_to_idx.items()}

# ---------- Load model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = EfficientNet_B0_Weights.IMAGENET1K_V1

mapping_path = MODEL_DIR / "class_to_idx.json"
weights_path = MODEL_DIR / "efficientnet_b0_plant_disease.pth"

if not mapping_path.exists() or not weights_path.exists():
    raise FileNotFoundError(f"Please place the model files in {MODEL_DIR!s}:\\n - class_to_idx.json\\n - efficientnet_b0_plant_disease.pth")

idx_to_class = load_idx_to_class(mapping_path)

model = efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(idx_to_class))
state = torch.load(weights_path, map_location=device)
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

# ---------- Grad-CAM ----------
def gradcam_for_image(model, tensor_img, target_idx=None):
    # Hooks to capture features and gradients
    features = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal features
        features = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    handle_f = model.features.register_forward_hook(forward_hook)
    handle_b = model.features.register_full_backward_hook(backward_hook)

    model.zero_grad()
    tensor_img = tensor_img.to(device).requires_grad_(True)
    logits = model(tensor_img)
    if target_idx is None:
        target_idx = int(logits.argmax(dim=1).item())

    loss = logits[:, target_idx].sum()
    loss.backward(retain_graph=True)

    pooled = gradients.mean(dim=[0,2,3], keepdim=True)  # [C,1,1]
    cam = (pooled * features).sum(dim=1, keepdim=True)  # [1,1,H,W]
    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()
    # normalize
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    handle_f.remove()
    handle_b.remove()
    return cam, target_idx

def overlay_cam_on_image(img_pil, cam):
    img = np.array(img_pil).astype(np.uint8)
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# ---------- Flask app ----------
app = Flask(__name__)
app.secret_key = "change-this-secret"

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        f = request.files["file"]
        if f.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if not allowed_file(f.filename):
            flash("Unsupported file type")
            return redirect(request.url)
        filename = secure_filename(f.filename)
        save_path = UPLOADS / filename
        f.save(save_path)
        return redirect(url_for("result", filename=filename))
    return render_template("index.html")

@app.route("/result/<filename>")
def result(filename):
    file_path = UPLOADS / filename
    if not file_path.exists():
        flash("File not found")
        return redirect(url_for("index"))
    # predict
    img_pil = Image.open(file_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = idx_to_class[pred_idx]
        conf = float(probs[pred_idx])
    # create grad-cam overlay
    cam, _ = gradcam_for_image(model, transform(img_pil).unsqueeze(0))
    overlay = overlay_cam_on_image(img_pil, cam)
    # save overlay to bytes for display
    overlay_path = UPLOADS / f"cam_{filename}"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return render_template("result.html", image=filename, cam_image=overlay_path.name, label=pred_class, conf=conf)

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error":"no file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error":"no file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error":"unsupported file type"}), 400
    filename = secure_filename(f.filename)
    save_path = UPLOADS / filename
    f.save(save_path)
    img_pil = Image.open(save_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        pred_idx = int(np.argmax(probs))
        pred_class = idx_to_class[pred_idx]
        conf = float(probs[pred_idx])
    return jsonify({"pred_class": pred_class, "confidence": conf, "probs": probs})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
