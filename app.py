import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
from YOLOv8.app_result.detect import run_YOLO_detection
# from YOLOv8.app_result.run_GradCAM import run_YOLO_gradcam
import tempfile
import nbformat
import os
from nbclient import NotebookClient


models = ["YOLOv8", "ResNet50", "See Both"]
img_types = ["jpg", "jpeg", "png"]
class_map = [1, 2, 5, 10]
coins  = ["1 Baht", "2 Baht", "5 Baht", "10 Baht"]
sections = ["Overview", "Feature map", "Grad-CAM"]

# **text** - bold text
# :color[text] - color text
st.title("Coin Counter and Calculator 🪙")
st.markdown("#### Upload a picture of :rainbow[coins] and let us handle counting for you!")
"A project of **Image Processing Course**. This project uses both YOLOv8 model and ResNet50 model for coin detection. We also provide a feature to count coins in an image including calculating the total value."
# st.html("<p>hi</p>")
st.markdown("source code [GitHub](%s)" % "https://github.com/jueeeeeen/Coin-Counter-and-Calculator-Model")
''

# st.subheader("1. Select a model to use")
# model = st.pills("the selected model will be used for predicting. If you choose both model we will show both model outputs for you.", models)
# ''
st.subheader("📷 Upload an image")
img_file = st.file_uploader(
    "We recommend taking picture clearly and using plain background with enough light for the best result!",
    img_types
    )
st.image(img_file, width=100)
''

st.subheader("👀 See the result")
'Please select a model and upload an image to proceed.'

btn_disable = False if img_file else True

st.button("Submit", type="primary", disabled=btn_disable)

if not btn_disable:
# if st.button("Submit", type="primary", disabled=btn_disable) and model:
    image = Image.open(img_file)
    result_img, counts, total_count = run_YOLO_detection(image)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()  # CLOSE IT → unlocks it on Windows

    # Now save to the path safely
    image.save(tmp_path)

    # Pass the path to your original Grad-CAM
    # gradcam_output = run_YOLO_gradcam(tmp_path)
    # Load the notebook
    nb = nbformat.read("./YOLOv8/app_result/grad_cam.ipynb", as_version=4)
    
    # Inject variables before execution
    for cell in nb.cells:
        if cell.cell_type == "code" and "# PARAMETERS" in cell.source:
            cell.source = f"""
    # PARAMETERS
    img_path = r"{tmp_path}"
    gradcam_output = None
    """

    # Create a NotebookClient
    client = NotebookClient(nb, kernel_name="python3")
    
    # Execute the notebook
    client.execute()
    
    gradcam_folder = "YOLOv8/app_result/gradcam_output"

    YOLO_tab, ResNet_tab = st.tabs(models[:2])

    with YOLO_tab:
        section = st.pills(None, options=sections, default=sections[0]) 
        total_value = np.sum(np.array(class_map) * np.array(counts))
        df = pd.DataFrame({"Coin": coins, "Count": counts})
        
        if section == sections[0]:
            each_coin_tab, total_coin_tab, total_value_tab = st.columns([5, 2, 3])
            with each_coin_tab:
                count_cols = st.columns(4, gap=None, border=True)
                for i, count_col in enumerate(count_cols):
                    count_col.metric(label=coins[i], value=counts[i])
                            
            total_coin_tab.metric(label="Total Coins", value=total_count, border=True)
            total_value_tab.metric(label="Total Value (Baht)", value=f"฿{total_value}", border=True)
            st.image(result_img, channels="BGR")
            
        elif section == section[1]:
            pass
        
        else:
            coin_1, coin_2, coin_5, coin_10 = st.columns(4)
            for class_name  in os.listdir(gradcam_folder):
                class_path = os.path.join(gradcam_folder, class_name)
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    st.image(img_path, caption=img_file)
                    
            coin_1.image("./YOLOv8/app_result/gradcam_output/1/3.jpg")
        
# st.code("for i in range(8): foo()")
# st.badge("New")
# st.html("<p>Hi!</p>")

# st.write("Most objects") # df, err, func, keras!
# st.latex(r""" e^{i\pi} + 1 = 0 """)

# # slider o------
# x = st.slider('x')
# st.write(x, 'squared is', x * x)

# # o---o
# st.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )

# # text input
# st.text_input("Your name", key="name")
# st.write(st.session_state.name) # access the value

# # multi select
# models = ["YOLOv8", "ResNet50"]
# model = st.multiselect("Select a model to try", models, default="YOLOv8")

# # toggle button
# rolling_average = st.toggle("Rolling average")

# # container
# with st.container(border=True):
#     st.write("I am in container")

# chart_data = pd.DataFrame(
#     np.random.randn(10, 3),
#     columns=['a', 'b', 'c'])

# # checkbox    
# if st.checkbox('Show dataframe'):
#     chart_data
    
# # tabs
# tab1, tab2 = st.tabs(["Chart", "Dataframe"])
# tab1.line_chart(chart_data, height=250)
# tab2.dataframe(chart_data, height=250, use_container_width=True)

# df = pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
#     })

# # dropdown
# option = st.selectbox(
#     'Which number do you like best?',
#     df['first column'])

# 'You selected: :D', option

# # side bar
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")
    
# latest_iteration = st.empty()
# bar = st.progress(0)
# for i in range(100):
#     # Update the progress bar with each iteration.
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i + 1)
#     time.sleep(0.1)

# '...and now we\'re done!'

# if st.button("Send balloons!"):
#     st.balloons()