import streamlit as st
from FasterRCNN.app_result.detect import run_fasterrcnn_detection
# from FasterRCNN.app_result.grad_cam import run_gradcam
import tempfile
import nbformat
import os
import numpy as np
from nbclient import NotebookClient

# @st.cache_data(show_spinner=False)
def run_fasterrcnn_result(_image):
    class_map = [1, 2, 5, 10]
    # class_map = ["__background__", "1", "2", "5", "10"]
    
    # Detection Overview
    faster_result_img, counts, total_count = run_fasterrcnn_detection(_image)
    total_value = np.sum(np.array(class_map) * np.array(counts))
    
    # GradCam - run jupyter notebook
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _image.save(tmp_path)

    nb = nbformat.read("FasterRCNN/app_result/grad_cam.ipynb", as_version=4)
    nb.cells[0].source = f"""
    # PARAMETERS
    img_path = r"{tmp_path}"
    gradcam_output = None
    """
    client = NotebookClient(nb, kernel_name="python3")
    client.execute()
    
    gradcam_folder = "FasterRCNN/app_result/gradcam_output"
    gradcam_grid = [[None for _ in range(4)] for _ in range(3)]
    class_to_col = {'1': 0, '2': 1, '5': 2, '10': 3}
    layer_to_row = {'layer2': 0, 'layer3': 1, 'layer4': 2}
    
    for class_name in os.listdir(gradcam_folder):
        class_path = os.path.join(gradcam_folder, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            for prefix, row_idx in layer_to_row.items():
                if img_file.startswith(prefix):
                    row = row_idx
                    break
            else:
                continue
            
            col = class_to_col.get(class_name)
            if col is not None:
                gradcam_grid[row][col] = img_path

    return faster_result_img, counts, total_count, total_value, gradcam_grid