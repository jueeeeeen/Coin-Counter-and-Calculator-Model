from YOLOv8.app_result.model_loader import model
from ultralytics.nn.modules.head import Head
import import_ipynb
from GradCam import get_grad_cam

# from model_loader import model

from ultralytics import YOLO
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# from PIL import Image


model.model = model.model.cuda()
# model.model.eval()
model.model.train()               # ✅ Use train() to allow gradients
for p in model.model.parameters():
    p.requires_grad_(True)
    
# Monkey-patch YOLO heads
for module in model.model.model:
    if isinstance(module, Head):
        if hasattr(module, "_forward"):
            module._inference = module._forward
        elif hasattr(module, "forward"):
            module._inference = module.forward

def preprocess_image(img_path, img_size=640):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    # Prepare tensor
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # HWC -> CHW
    img = img.unsqueeze(0).to(next(model.model.parameters()).device)  # [1,C,H,W]

    return img

def register_gradcam_hooks(target_layer):
    """
    Register forward and backward hooks on a given layer.
    Returns:
        activations: dict to store forward output
        gradients: dict to store backward gradients
        handles: tuple of (forward_handle, backward_handle) for later removal
    """
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    return activations, gradients, (fwd_handle, bwd_handle)

def apply_heatmap_overlay(img, heatmap):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    return overlay


def calculate_grad_cam(score_tensor, shape, activations, gradients):
    # Backward pass to get gradients
    model.model.zero_grad()
    # score_tensor = score_tensor.clone().detach().requires_grad_(True)
    score_tensor.backward(retain_graph=True)    

    # Grad-CAM computation
    grad = gradients['value'][0]  # [C,H,W]
    act = activations['value'][0] # [C,H,W]
    
    weights = grad.mean(dim=(1,2), keepdim=True)  # global average pooling
    
    cam = (weights * act).sum(dim=0)  # sum over channels
    
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam_np = cam.detach().cpu().numpy()
        
    cam_np = cv2.resize(cam_np, (shape[1], shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255*cam_np), cv2.COLORMAP_JET)
        
    return heatmap

def get_layer(layer_target):
    # Pick target layer
    if isinstance(layer_target, int):
        return  model.model.model[layer_target]
    elif isinstance(layer_target, tuple) and len(layer_target) == 2:
        layer_idx, sublayer_name = layer_target
        return  dict(model.model.model[layer_idx].named_modules())[sublayer_name]
    else:
        raise ValueError("layer_target must be int or (int, str)")
    
def get_boxes_score(img_tensor, class_indices):
    img_tensor = img_tensor.cuda()
    
    img_tensor.requires_grad_(True)
    
    # Forward pass through backbone to trigger hooks
    # outputs = model.model(img_tensor)
    with torch.enable_grad():        # <---- REQUIRED
        outputs = model.model(img_tensor)

    # Use objectness of the highest-scoring predicted box 
    pred_tensor = outputs[0][0]
    """
    shape: (8400, 8)
    1 anchor: (x, y, w, h, class-1, class-10, class-2, class-5)
    """
    
    # get highest class score boxes -------------------------------
    boxes = []
    
    for i in range(class_indices[0], class_indices[1]):
        top_idx = pred_tensor[4 + i, :].argmax()
        box_tensor = pred_tensor[:, top_idx]
        boxes.append({
            "class": model.names[i],
            "idx": top_idx,
            "score": box_tensor[4 + i],
            "coordinates": box_tensor[:4].tolist()
        })
    # ---------------------------------------------------\
    return boxes

def get_grad_cam(img_path, layer_target, class_indices = [0, 4]):
    """
    layer_target can be:
        - int: top-level layer index
        - tuple: (layer_index, sublayer_name) for nested submodules
    """
    
    layer = get_layer(layer_target)
    
    # Register hooks
    activations, gradients, handles = register_gradcam_hooks(layer)

    # Read and preprocess image
    img = cv2.imread(img_path)
    # img = np.array(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_tensor = preprocess_image(img_path)
    
    boxes = get_boxes_score(img_tensor, class_indices)
    
    output_images = []
    
    # Calculate Grad-CAM
    for box in boxes:
        heatmap = calculate_grad_cam(box["score"], img.shape[:2], activations, gradients)        
        # Overlay heatmap
        output_image = apply_heatmap_overlay(img, heatmap)
        
        layer_name = "-".join(map(str, layer_target)) if isinstance(layer_target, tuple) else str(layer_target)
        save_path = f'GradCam/correct/separated/{box["class"]}/{layer_name}.jpg'
        cv2.imwrite(save_path, output_image)
        
        output_images.append(output_image)
        
    # Remove hooks
    handles[0].remove()
    handles[1].remove()

    return output_images, boxes

def run_YOLO_gradcam(img):
    layers = [3, (12, 'cv2'), (18, 'cv1')]
    output = []
    for i, layer in enumerate(layers):
        output_images, boxes = get_grad_cam(img, layer)
        output.append(output_images)
        # for j, img in enumerate(output_images[:images_per_layer]):
        #     axes[i][j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     axes[i][j].set_title(f"{boxes[j]["class"]}-Layer {layer}" if isinstance(layer, int) else f"{boxes[j]["class"]}-Layer {layer[0]}:{layer[1]}")
        #     axes[i][j].axis('off')
    return output
# img_file = "test_images/test_1.jpg"
# img = Image.open(img_file)
# run_YOLO_gradcam(img)