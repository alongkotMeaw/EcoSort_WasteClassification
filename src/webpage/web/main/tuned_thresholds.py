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
    "web_meaw/model/best_model_fold1.pth",
    "web_meaw/model/best_model_fold2.pth",
    "web_meaw/model/best_model_fold3.pth",
    "web_meaw/model/best_model_fold4.pth",
    "web_meaw/model/best_model_fold5.pth",
]

class_names = [
    "AluminumCan", "Battery", "Cardboard", "ChargerCable", "Earphones",
    "FoamContainer", "FoodWaste", "GlassBottle", "LightBulb", "Paper",
    "PesticideContainer", "Phone", "PlasticBottle", "PlasticCup",
    "PowerBank", "SnackWrapper", "SprayCan", "Straw", "TissuePaper",
    "YardWaste", "Toothbrush"
]

# --- 5 หมวดหลัก ---
GROUPS = {
    "Recyclable": {
        "label_th": "ขยะรีไซเคิล (Recyclable)",
        "members": [
            "PlasticBottle", "AluminumCan", "PlasticCup",
            "Cardboard", "Paper"
        ]
    },
    "General": {
        "label_th": "ขยะทั่วไป (General Waste)",
        "members": [
            "SnackWrapper", "Straw", "FoamContainer", "Toothbrush"
        ]
    },
    "Biodegradable": {
        "label_th": "ขยะย่อยสลายได้ (Biodegradable)",
        "members": [
            "FoodWaste", "YardWaste", "TissuePaper"
        ]
    },
    "Hazardous": {
        "label_th": "ขยะอันตราย (Hazardous)",
        "members": [
            "Battery", "SprayCan", "PesticideContainer", "LightBulb"
        ]
    },
    "Ewaste": {
        "label_th": "ขยะอิเล็กทรอนิกส์ (E-Waste)",
        "members": [
            "Phone", "ChargerCable", "Earphones", "PowerBank"
        ]
    }
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_FILE = PROJECT_ROOT / "Web" / "class_thresholds.json"

# ===== THRESHOLD =====
_STATIC_THRESHOLD_PERCENT = {
    "PlasticCup": 3.0,
    "Battery": 90.0,
    "Paper": 70.0,
    "PlasticBottle": 12.0,
    "GlassBottle": 30.0,
    "YardWaste": 18.0,
}
DEFAULT_THRESHOLD = 50.0  # percentage


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

# ===== MODEL =====
def load_resnet_model(path):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

models_ensemble = [load_resnet_model(p) for p in MODEL_PATHS]

# ===== PREPROCESS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

app = Flask(__name__)

# ===== HELPER =====
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

def group_results(result_dict: dict[str, float]):
    th = {c: get_class_threshold(c) for c in result_dict}
    groups_render = []
    for key, meta in GROUPS.items():
        items = []
        for m in meta["members"]:
            if m not in result_dict: 
                continue
            p = float(result_dict[m])
            t = float(th[m])
            items.append({
                "name": m,
                "prob": p,
                "prob_pct": f"{p*100:.2f}",
                "thresh": t,
                "thresh_pct": f"{t*100:.1f}",
                "pass": p >= t
            })
        items.sort(key=lambda x: x["prob"], reverse=True)
        groups_render.append({"title": meta["label_th"], "items": items})
    return groups_render

# ===== ROUTE =====
@app.route("/", methods=["GET","POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", prediction=None, error="กรุณาเลือกไฟล์ภาพก่อน")

        filename = secure_filename(file.filename)
        save_dir = Path("static") / "uploads"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        file.save(save_path)

        img = Image.open(save_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        probs = predict_ensemble(img_tensor)
        result_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        groups_render = group_results(result_dict)

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

        percent_display = {k: f"{v*100:.2f}%" for k,v in result_dict.items()}
        sorted_result = dict(sorted(percent_display.items(), key=lambda x: float(x[1][:-1]), reverse=True))

        return render_template("index.html",
            prediction=final_class,
            img_path=f"uploads/{filename}",
            probs=sorted_result,
            top_label=final_class,
            top_prob=f"{final_prob*100:.2f}%",
            threshold=f"{threshold*100:.1f}%",
            chart_labels=list(sorted_result.keys()),
            chart_values=[float(v[:-1]) for v in sorted_result.values()],
            groups=groups_render
        )
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\nLocal:   http://127.0.0.1:5000/")
    print(f"Network: http://{local_ip}:5000/\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
