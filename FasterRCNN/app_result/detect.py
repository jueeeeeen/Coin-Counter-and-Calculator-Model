from FasterRCNN.app_result.model_loader import model
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
import torch
import numpy as np
from PIL import Image


def filter_nms(pred_boxes, pred_labels, pred_scores, iou_threshold, score_threshold):
    # filter low score boxes
    mask = pred_scores > score_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    #  nms to remove overlapping boxes
    mask = nms(pred_boxes, pred_scores, iou_threshold)
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]
    return pred_boxes, pred_labels, pred_scores

def draw_bbox_jaaa(img,  pred_boxes, pred_labels, pred_scores, classes):
    # img = torch.tensor(np.array(img)).permute(2, 0, 1).to(device)
    # img_cuda = img.float() / 255.0
    # output = model([img_cuda])[0]
    class_colors = {
        "__background__": "black",
        "1": "green",
        "10": "orange",
        "2": "blue",
        "5": "red",
    }
    box_colors = [class_colors[classes[int(lbl)]] for lbl in pred_labels]

    # labels with confidence scores
    labels_with_conf = [f"{classes[int(lbl)]}: {float(score):.2f}" for lbl, score in zip(pred_labels, pred_scores)]

    img_int = (img.cpu() * 255).to(torch.uint8)
    return draw_bounding_boxes(
        img_int, pred_boxes, labels_with_conf, width=2, colors=box_colors, font_size=25, font="FasterRCNN/Kanit-Black.ttf"
    ).permute(1, 2, 0)

# result = draw_bbox_jaaa(img, model, device, ["__background__", "1", "10", "2", "5"])

def run_fasterrcnn_detection(img):  
    img = torch.tensor(np.array(img)).permute(2, 0, 1).float().to(next(model.parameters()).device) / 255.0
    classes = ["__background__", "1", "10", "2", "5"]
    preds = model([img])[0]
    pred_boxes = preds["boxes"].cpu()
    pred_labels = preds["labels"].cpu()
    pred_scores = preds["scores"].cpu()
    
    pred_boxes, pred_labels, pred_scores = filter_nms(
        pred_boxes, pred_labels, pred_scores, iou_threshold=0.5, score_threshold=0.5
    )

    result_img = draw_bbox_jaaa(img, pred_boxes, pred_labels, pred_scores, classes)
    result_img = Image.fromarray(result_img.numpy())

    counts = [
        (pred_labels == 1).sum(),   
        (pred_labels == 3).sum(),
        (pred_labels == 4).sum(),
        (pred_labels == 2).sum(),
    ]
    total_count = len(pred_labels)
    
    return result_img, counts, total_count

# def run_fasterrcnn_detection(img):
#     preds = model(img)
#     result_img = preds[0].plot()
    
#     detected_cls = preds[0].boxes.cls.cpu().tolist()
#     counts = [detected_cls.count(0), detected_cls.count(2), detected_cls.count(3), detected_cls.count(1)]
    
#     total_count = len(preds[0].boxes)
    
#     return result_img, counts, total_count

