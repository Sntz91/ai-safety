import argparse
import os
import glob
import yaml
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ai_safety.models.diagnostic.vit import ViT
from ai_safety.data.transforms import Transform

# ----------------- CONFIG -----------------
WEIGHTS = "runs/diagnostic/3083/best_model.pt"
THRESHOLD = "runs/diagnostic/3083/thresholds.yaml"
# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = Transform(train=False, image_size=224)

# Load model & thresholds
ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ViT(num_classes=state["head.weight"].shape[0]).to(device)
model.load_state_dict(state)
model.eval()

with open(THRESHOLD) as f:
    thresholds = yaml.safe_load(f)

def predict(path):
    ds = pydicom.dcmread(path)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = ds.pixel_array.astype(np.float32) * slope + intercept
    tensor = transform(hu).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).cpu().numpy().item()
    disp_img = np.clip((hu - 0.0) / 80.0, 0.0, 1.0)
    return prob, disp_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diagnostic Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to DICOM file or folder")
    args = parser.parse_args()
    input_path = args.input

    if os.path.isdir(input_path):
        def sort_key(f):
            try:
                return int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber)
            except Exception:
                return f

        files = sorted([f for f in glob.glob(os.path.join(input_path, "*")) if os.path.isfile(f)], key=sort_key)
        results = [predict(f) for f in files]
        probs, imgs = [r[0] for r in results], [r[1] for r in results]

        prob = float(np.mean(np.sort(probs)[-min(3, len(probs)):]))
        thresh = thresholds["scan"][0]
        decision = "POSITIVE" if prob >= thresh else "NEGATIVE"
        print(f"Scan Prediction: {prob:.4f} | Thresh: {thresh:.4f} | Decision: {decision}")

        # Scan Viz with Slider
        fig, ax = plt.subplots(figsize=(6, 7))
        plt.subplots_adjust(bottom=0.15)
        im = ax.imshow(imgs[0], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Scan: {prob:.4f} | Thresh: {thresh:.4f} | {decision}\nSlice [0/{len(imgs)-1}] Prob: {probs[0]:.4f}")

        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, "Slice", 0, len(imgs) - 1, valinit=0, valstep=1)

        def update(val):
            idx = int(slider.val)
            im.set_data(imgs[idx])
            ax.set_title(f"Scan: {prob:.4f} | Thresh: {thresh:.4f} | {decision}\nSlice [{idx}/{len(imgs)-1}] Prob: {probs[idx]:.4f}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

    else:
        prob, img = predict(input_path)
        thresh = thresholds["slice"][0]
        decision = "POSITIVE" if prob >= thresh else "NEGATIVE"
        print(f"Slice Prediction: {prob:.4f} | Thresh: {thresh:.4f} | Decision: {decision}")

        # Slice Viz
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="gray")
        plt.title(f"Slice Prediction: {prob:.4f} | Thresh: {thresh:.4f} | {decision}")
        plt.axis("off")
        plt.show()
