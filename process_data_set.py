import cv2
import numpy as np
import os

input_folder = "C:/Users/Jin/Downloads/5"
output_folder = "dataset"
target_size = (2048, 2048)

os.makedirs(output_folder, exist_ok=True)

for i, filename in enumerate(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        new_name = f"five_{i + 6:0>5}_aug0.jpg"
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # คำนวณ scale factor
        scale = min(target_size[0]/w, target_size[1]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (new_w, new_h))

        # คำนวณสีเฉลี่ย
        mean_color = cv2.mean(resized)[:3]
        canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8)
        canvas[:] = mean_color  # เติมด้วยสีเฉลี่ย

        # วางรูปกลาง canvas
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        save_path = os.path.join(output_folder, new_name)
        cv2.imwrite(save_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])