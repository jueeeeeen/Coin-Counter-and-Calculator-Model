import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")

target_layers = [0, 2, 4]
feature_maps = {}

def hook_fmap(module, input, output):
    feature_maps[module] = output.detach()

for i, layer in enumerate(model.model.model):
    if i in target_layers:
        layer.register_forward_hook(hook_fmap)

def save_feature_maps(_image, save_dir="featuremaps"):
    feature_maps.clear()

    os.makedirs(save_dir, exist_ok=True)

    if isinstance(_image, str):
        img_input = _image
    elif isinstance(_image, Image.Image):
        img_input = np.array(_image)
    elif isinstance(_image, np.ndarray):
        img_input = _image
    else:
        raise TypeError(f"Unsupported image type: {type(_image)}")

    _ = model(img_input)

    saved_paths = []
    for idx, (layer, fmap) in enumerate(feature_maps.items()):
        fmap = fmap[0]
        fmap_np = fmap.cpu().numpy()
        mean_map = np.mean(fmap_np, axis=0)

        plt.figure(figsize=(5, 5))
        plt.imshow(mean_map)
        plt.axis('off')
        
        save_path = os.path.abspath(os.path.join(save_dir, f"Layer {target_layers[idx] + 1}.png"))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        saved_paths.append(save_path)

    return saved_paths
