import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from FasterRCNN.app_result.model_loader import model
from torchvision import transforms

# ---------- เลือก target layers ----------
# backbone.body = ResNet;  layer2, layer3, layer4 เหมาะสุด
# target_layers = [
#     model.backbone.body.layer2,
#     model.backbone.body.layer3,
#     model.backbone.body.layer4
# ]
target_layers = {
    "layer2": model.backbone.body.layer2,
    "layer3": model.backbone.body.layer3,
    "layer4": model.backbone.body.layer4,
}

feature_maps = {}

def hook_fmap(name):
    # บันทึก feature map ออกมาเก็บใน dict
    # feature_maps[module] = output.detach()
    def hook(module, input, output):
        feature_maps[name] = output.detach()
    return hook

# ลงทะเบียน hook
for name, layer in target_layers.items():
    layer.register_forward_hook(hook_fmap(name))

# ---------- ฟังก์ชันหลัก ----------
def save_feature_maps(_image, save_dir="featuremaps"):
    device = next(model.parameters()).device

    feature_maps.clear()
    os.makedirs(save_dir, exist_ok=True)

    # รองรับ input ได้ทั้ง path, PIL, numpy
    if isinstance(_image, str):
        img = Image.open(_image).convert("RGB")
    elif isinstance(_image, Image.Image):
        img = _image.convert("RGB")
    elif isinstance(_image, np.ndarray):
        img = Image.fromarray(_image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(_image)}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).to(device)

    # รันโมเดล
    _ = model([img_tensor])

    saved_paths = []
    for name, fmap in feature_maps.items():
        fmap = fmap[0]
        fmap_np = fmap.cpu().numpy()
        mean_map = np.mean(fmap_np, axis=0)
        plt.imshow(mean_map, cmap="viridis")
        plt.axis("off")
        save_path = os.path.join(save_dir, f"{name}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        saved_paths.append(save_path)

    return saved_paths
