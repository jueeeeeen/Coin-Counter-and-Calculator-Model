from YOLOv8.app_result.model_loader import model

def run_YOLO_detection(img):
    preds = model(img)
    result_img = preds[0].plot()
    
    detected_cls = preds[0].boxes.cls.cpu().tolist()
    counts = [detected_cls.count(0), detected_cls.count(2), detected_cls.count(3), detected_cls.count(1)]
    
    total_count = len(preds[0].boxes)
    
    return result_img, counts, total_count

