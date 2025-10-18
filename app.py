import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
from YOLOv8.app_result.all_result import run_yolo_result
from FasterRCNN.app_result.all_result import run_fasterrcnn_result
import os


models = ["YOLOv8", "FasterRCNN", "See Both"]
img_types = ["jpg", "jpeg", "png"]
class_map = [1, 2, 5, 10]
coins  = ["1 Baht", "2 Baht", "5 Baht", "10 Baht"]
sections = ["Overview", "Feature map", "Grad-CAM"]

# **text** - bold text
# :color[text] - color text
st.title("Coin Counter and Calculator 🪙")
st.markdown("#### Upload a picture of :rainbow[coins] and let us handle counting for you!")
"A project of **Image Processing Course**. This project uses both YOLOv8 model and FasterRCNN model for coin detection. We also provide a feature to count coins in an image including calculating the total value."
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
if img_file:
    st.image(img_file, width=100)
''

st.subheader("👀 See the result")
if not img_file:
    ':red[Please upload an image to proceed.]'

submit_btn = st.button("Submit", type="primary", disabled=(img_file == None))
loading_placeholder = st.empty()
YOLO_tab, RCNN_tab = st.tabs(models[:2])
yolo_section_pills = YOLO_tab.pills(None, options=sections, default=sections[0], key="yolo")
yolo_section = YOLO_tab.empty()

rcnn_section_pills = RCNN_tab.pills(None, options=sections, default=sections[0], key="rcnn")
rcnn_section = RCNN_tab.empty()

# if not btn_disable:
if submit_btn:
    st.session_state.clear()
    # open uploaded image
    image = Image.open(img_file)
    
    # START: YOLOv8 model ----------------------------------------------------
    with loading_placeholder.container():
        with st.spinner("Processing image and generating Result"):
            yolo_result_img, counts, total_count, total_value, gradcam = run_yolo_result(image)
            st.session_state["yolo_result_img"] = yolo_result_img
            st.session_state["yolo_counts"] = counts
            st.session_state["yolo_total_count"] = total_count
            st.session_state["yolo_total_value"] = total_value
            st.session_state["yolo_gradcam"] = gradcam
    
    # END: YOLOv8 model ----------------------------------------------------

    # START: FasterRCNN model ----------------------------------------------------
    with loading_placeholder.container():
        with st.spinner("Processing image and generating Result"):
            rcnn_result_img, counts, total_count, total_value, gradcam = run_fasterrcnn_result(image)
            st.session_state["rcnn_result_img"] = rcnn_result_img
            st.session_state["rcnn_counts"] = counts
            st.session_state["rcnn_total_count"] = total_count
            st.session_state["rcnn_total_value"] = total_value
            st.session_state["rcnn_gradcam"] = gradcam

    # END: FasterRCNN model ----------------------------------------------------


if "yolo_gradcam" in st.session_state:
    with YOLO_tab:
        result_img = st.session_state["yolo_result_img"]
        counts = st.session_state["yolo_counts"]
        total_count = st.session_state["yolo_total_count"]
        total_value = st.session_state["yolo_total_value"]
        gradcam = st.session_state["yolo_gradcam"]
                
        if yolo_section_pills == sections[0]:
            with yolo_section.container():
                each_coin_tab, total_coin_tab, total_value_tab = st.columns([5, 2, 3])
                with each_coin_tab:
                    count_cols = st.columns(4, gap=None, border=True)
                    for i, count_col in enumerate(count_cols):
                        count_col.metric(label=coins[i], value=counts[i])
                                
                total_coin_tab.metric(label="Total Coins", value=total_count, border=True)
                total_value_tab.metric(label="Total Value (Baht)", value=f"฿{total_value}", border=True)
                st.image(result_img, channels="BGR")
            
        elif yolo_section_pills == sections[1]:
            with yolo_section.container():
                pass
        
        else:
            with yolo_section.container():
                column_header = st.columns([1.3, 2, 2, 2, 2])
                for i in range(4):
                    column_header[i+1].write(coins[i])
                layers = [st.columns([1.3, 2, 2, 2, 2]), st.columns([1.3, 2, 2, 2, 2]), st.columns([1.3, 2, 2, 2, 2])]
                layers[0][0].write("layer 3")
                layers[1][0].write("layer 12-cv2")
                layers[2][0].write("layer 18-cv1")
                for i, row in enumerate(gradcam):
                    for col, img_path in zip(layers[i][1:], row):
                        col.image(img_path)

if "rcnn_gradcam" in st.session_state:
    with RCNN_tab:
        result_img = st.session_state["rcnn_result_img"]
        counts = st.session_state["rcnn_counts"]
        total_count = st.session_state["rcnn_total_count"]
        total_value = st.session_state["rcnn_total_value"]
        gradcam = st.session_state["rcnn_gradcam"]

        if rcnn_section_pills == sections[0]:
            with rcnn_section.container():
                each_coin_tab, total_coin_tab, total_value_tab = st.columns([5, 2, 3])
                with each_coin_tab:
                    count_cols = st.columns(4, gap=None, border=True)
                    for i, count_col in enumerate(count_cols):
                        count_col.metric(label=coins[i], value=counts[i])
                                
                total_coin_tab.metric(label="Total Coins", value=total_count, border=True)
                total_value_tab.metric(label="Total Value (Baht)", value=f"฿{total_value}", border=True)
                st.image(result_img, channels="BGR")

        elif rcnn_section_pills == sections[1]:
            with rcnn_section.container():
                pass
        
        else:
            with rcnn_section.container():
                column_header = st.columns([1.3, 2, 2, 2, 2])
                for i in range(4):
                    column_header[i+1].write(coins[i])
                layers = [st.columns([1.3, 2, 2, 2, 2]), st.columns([1.3, 2, 2, 2, 2]), st.columns([1.3, 2, 2, 2, 2])]
                layers[0][0].write("layer 3")
                layers[1][0].write("layer 12-cv2")
                layers[2][0].write("layer 18-cv1")
                for i, row in enumerate(gradcam):
                    for col, img_path in zip(layers[i][1:], row):
                        col.image(img_path)


    # with RCNN_tab:
    #     # result_img = st.session_state["resnet_result_img"]
    #     # counts = st.session_state["resnet_counts"]
    #     # total_count = st.session_state["resnet_total_count"]
    #     # total_value = st.session_state["resnet_total_value"]
    #     # gradcam = st.session_state["resnet_gradcam"]
        
    #     if rcnn_section_pills == sections[0]:
    #         with rcnn_section.container():
    #             each_coin_tab, total_coin_tab, total_value_tab = st.columns([5, 2, 3])
    #             with each_coin_tab:
    #                 count_cols = st.columns(4, gap=None, border=True)
    #                 for i, count_col in enumerate(count_cols):
    #                     count_col.metric(label=coins[i], value=counts[i])
                                
    #             total_coin_tab.metric(label="Total Coins", value=total_count, border=True)
    #             total_value_tab.metric(label="Total Value (Baht)", value=f"฿{total_value}", border=True)
    #             st.image(result_img, channels="BGR")

    #     elif rcnn_section_pills == sections[1]:
    #         with rcnn_section.container():
    #             pass
        
    #     else:
    #         with rcnn_section.container():
    #             column_header = st.columns([1.3, 2, 2, 2, 2])
    #             for i in range(4):
    #                 column_header[i+1].write(coins[i])
    #             layers = [st.columns([1.3, 2, 2, 2, 2]), st.columns([1.3, 2, 2, 2, 2]), st.columns([1.3, 2, 2, 2, 2])]
    #             layers[0][0].write("layer 3")
    #             layers[1][0].write("layer 12-cv2")
    #             layers[2][0].write("layer 18-cv1")
    #             for i, row in enumerate(gradcam):
    #                 for col, img_path in zip(layers[i][1:], row):
    #                     col.image(img_path)
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