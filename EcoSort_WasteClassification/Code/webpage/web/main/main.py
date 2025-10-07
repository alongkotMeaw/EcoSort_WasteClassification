from flask import Flask, request, render_template
import json
import socket
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from werkzeug.utils import secure_filename

# ===== CONFIG =====
NUM_CLASSES = 21
MODEL_PATHS = [
    "web/model/best_model_fold1.pth",
    "web/model/best_model_fold2.pth",
    "web/model/best_model_fold3.pth",
    "web/model/best_model_fold4.pth",
    "web/model/best_model_fold5.pth",
]

class_names = [
    "AluminumCan", "Battery", "Cardboard", "ChargerCable", "Earphones",
    "FoamContainer", "FoodWaste", "GlassBottle", "LightBulb", "Paper",
    "PesticideContainer", "Phone", "PlasticBottle", "PlasticCup",
    "PowerBank", "SnackWrapper", "SprayCan", "Straw", "TissuePaper",
    "YardWaste", "Toothbrush"
]

# ===== MAIN GROUPS =====
GROUPS = {
    "Recyclable": {
        "label": "♻️ Recyclable Waste",
        "members": ["PlasticBottle", "AluminumCan", "PlasticCup", "Cardboard", "Paper"]
    },
    "General": {
        "label": "🗑️ General Waste",
        "members": ["SnackWrapper", "Straw", "FoamContainer", "Toothbrush"]
    },
    "Biodegradable": {
        "label": "🌱 Biodegradable Waste",
        "members": ["FoodWaste", "YardWaste", "TissuePaper"]
    },
    "Hazardous": {
        "label": "☣️ Hazardous Waste",
        "members": ["Battery", "SprayCan", "PesticideContainer", "LightBulb"]
    },
    "Ewaste": {
        "label": "📱 Electronic Waste (E-Waste)",
        "members": ["Phone", "ChargerCable", "Earphones", "PowerBank"]
    }
}

PROJECT_ROOT = Path(__file__).resolve().parents[0]
THRESHOLD_FILE = PROJECT_ROOT / "class_thresholds.json"
INFO_DIR = PROJECT_ROOT / "info_txt"

# ===== Threshold =====
_STATIC_THRESHOLD_PERCENT = {
    "PlasticCup": 3.0,
    "Battery": 90.0,
    "Paper": 70.0,
    "PlasticBottle": 12.0,
    "GlassBottle": 30.0,
    "YardWaste": 18.0,
}
DEFAULT_THRESHOLD = 50.0


def _load_tuned_thresholds() -> dict[str, float]:
    if not THRESHOLD_FILE.exists():
        return {}
    try:
        payload = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    thresholds = payload.get("thresholds")
    names = payload.get("class_names", [])
    tuned = {}
    if isinstance(thresholds, dict):
        for k, v in thresholds.items():
            try:
                tuned[k.lower()] = float(v)
            except:
                continue
    elif isinstance(thresholds, list) and len(thresholds) == len(names):
        for n, v in zip(names, thresholds):
            try:
                tuned[n.lower()] = float(v)
            except:
                continue
    return tuned


_FALLBACK_THRESHOLDS = {k.lower(): v / 100.0 for k, v in _STATIC_THRESHOLD_PERCENT.items()}
TUNED_THRESHOLDS = _load_tuned_thresholds()


# ===== Load model =====
def load_resnet_model(path):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


models_ensemble = [load_resnet_model(p) for p in MODEL_PATHS]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = Flask(__name__)


# ===== Helpers =====
def get_class_threshold(cname: str) -> float:
    key = cname.lower()
    if key in TUNED_THRESHOLDS:
        return TUNED_THRESHOLDS[key]
    if key in _FALLBACK_THRESHOLDS:
        return _FALLBACK_THRESHOLDS[key]
    return DEFAULT_THRESHOLD / 100.0


def predict_ensemble(img_tensor: torch.Tensor) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for m in models_ensemble:
            preds.append(F.softmax(m(img_tensor), dim=1).cpu().numpy())
    return np.mean(preds, axis=0)[0]


def find_bin_group_and_image(final_class: str):
    for gname, meta in GROUPS.items():
        if final_class in meta["members"]:
            bin_file = f"wasebin/{gname}.png"
            return meta["label"], bin_file, gname
    return "Unknown", None, None


def load_info_from_txt(group_key: str) -> str:
    if not group_key:
        return "No information available for this category."
    file_path = INFO_DIR / f"{group_key}.txt"
    if not file_path.exists():
        return "No additional data found for this waste type."
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading info: {e}"


# ===== Routes =====
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", prediction=None, error="Please select an image first.")
        filename = secure_filename(file.filename)
        save_dir = Path("data") / "uploads"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        file.save(save_path)

        img = Image.open(save_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        probs = predict_ensemble(img_tensor)
        result_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        candidates = []
        for c, p in result_dict.items():
            t = get_class_threshold(c)
            if p >= t:
                candidates.append((c, p, t))

        if candidates:
            final_class, final_prob, threshold = max(candidates, key=lambda x: x[1])
        else:
            final_class = max(result_dict, key=result_dict.get)
            final_prob = result_dict[final_class]
            threshold = get_class_threshold(final_class)

        bin_group_label, bin_image, group_key = find_bin_group_and_image(final_class)
        info_text = load_info_from_txt(group_key)

        return render_template("index.html",
                               prediction=final_class,
                               bin_group_label=bin_group_label,
                               bin_image=bin_image,
                               info_text=info_text)

    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    import subprocess
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\nLocal: http://127.0.0.1:5000/")
    print(f"Network: http://{local_ip}:5000/\n")
    try:
        ngrok = subprocess.Popen(["ngrok", "http", "5000"])
        print("Ngrok started. Check https://dashboard.ngrok.com to see the HTTPS URL.")
    except FileNotFoundError:
        print("ngrok.exe not found. Please install it first.")
    app.run(host="0.0.0.0", port=5000, debug=True)
