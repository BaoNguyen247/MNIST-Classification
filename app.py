import streamlit as st
import numpy as npS
import plotly.express as px
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import cv2
import pickle
import json


# Load your SVM model with st.cache
@st.cache_data
def load_model():
    model_file = open("results_svm\specify_rbf\svm_model_rbf_C4.pickle", "rb")
    return pickle.load(model_file)


st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <h1 style="text-align: center;">SVM Digit Recognition</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

section = st.sidebar.selectbox(
    "Select section",
    (
        "General Infomation",
        "Change parameter",
        "Detect handwrite digit",
        "Upload Image",
    ),
)


def add_parameter_gui(section):
    params = dict()
    if section == "Change parameter":
        kernel_selector = st.sidebar.selectbox(
            "Select Kernel", ["Radial Basis Function", "Polynomial"]
        )
        params["kernel_selector"] = kernel_selector
        if kernel_selector == "Radial Basis Function":
            params["C"] = st.sidebar.slider("C", 1, 12, step=1)
        elif kernel_selector == "Polynomial":
            params["Degree"] = st.sidebar.slider("Degree", 1, 10, step=1)
    return params


val = add_parameter_gui(section)

if section == "General Infomation":
    st.header("What is SVM ?")
    # Explanation of SVM
    st.markdown(
        """
        Support Vector Machines (SVM) are a class of supervised machine learning algorithms used for 
        classification and regression tasks.         
        ![Example Digit](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png)

        The key idea behind SVM is to find a hyperplane that best separates the data points into different 
        classes while maximizing the margin between them. 

        SVM is known for its effectiveness in solving both linear and non-linear classification problems, 
        making it a powerful tool in the field of machine learning.
        """
    )

    st.header("What is MNIST Dataset")

    # Explanation of the MNIST dataset
    st.markdown(
        """
        The MNIST dataset is a widely used dataset in the machine learning community. It consists of a 
        collection of 28x28 pixel grayscale images of handwritten digits (0-9). MNIST is a popular 
        benchmark dataset for tasks related to digit recognition and image classification.
        """
    )

    # Visualizations or images related to SVM and MNIST can be added here
    # For example, you can include sample images of handwritten digits or SVM decision boundaries

    st.header("Example:")

    # Explanation and example of SVM digit recognition
    st.markdown(
        """
        Let's illustrate how Support Vector Machines (SVM) can be applied to the MNIST dataset for digit 
        recognition. Suppose we have a handwritten digit image like the one shown below:

        ![Example Digit](https://www.researchgate.net/publication/237452158/figure/fig1/AS:298786795606021@1448247731187/A-sample-of-the-handwritten-digit-database-due-to-Guyon-et-al-1989-The-digits-have.png)

        Using SVM, we can train a model to recognize the digit in this image. The SVM algorithm learns to 
        distinguish the unique patterns and features of each digit, allowing it to make accurate predictions. 
        In this case, the SVM model would correctly identify the digit as "3."

        SVM's ability to classify handwritten digits is a classic example of its effectiveness in solving 
        real-world classification problems.
        """
    )


# Show the selected parameter
if section == "Change parameter":
    with open("results_svm/record.json", "r") as json_file:
        data = json.load(json_file)

    # Create separate DataFrames for rbf and poly kernels
    df_rbf = pd.DataFrame(data[0])  # Assumes rbf is the first entry in the JSON data
    df_poly = pd.DataFrame(data[1])  # Assumes poly is the second entry in the JSON data

    kernel_selector = val.get(
        "kernel_selector", "Radial Basis Function"
    )  # Get the selected kernel

    if kernel_selector == "Radial Basis Function":
        # Extract the relevant scores for the selected C value for rbf kernel
        selected_config = f"SVM _ Kernel : Radial Basis Function (RBF) _ C: {val['C']}"
        st.markdown(
            f'<p style="font-size: 24px;"><strong>{selected_config}</strong></p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size: 20px;"><strong>Normal Accuracy:  </strong> {(df_rbf["acc"].iloc[val["C"] - 1]) * 100:.2f}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size: 20px;"><strong>F1 Score: </strong>{(df_rbf["f1"].iloc[val["C"] - 1]) * 100:.2f}</p>',
            unsafe_allow_html=True,
        )

        # Create a line chart for rbf kernel
        fig = px.line(df_rbf, x="c", y=["acc", "f1"], markers=True, title="RBF Kernel")

    elif kernel_selector == "Polynomial":
        # Extract the relevant scores for the selected Degree value for poly kernel
        selected_config = f"SVM _ Kernel : Polynomial (poly) _ Degree: {val['Degree']}"

        st.markdown(
            f'<p style="font-size: 24px;"><strong>{selected_config}</strong></p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size: 20px;"><strong>Normal Accuracy: </strong>{df_poly["acc"].iloc[val["Degree"] - 1]* 100:.2f}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size: 20px;"><strong>F1 Score: </strong>{df_poly["f1"].iloc[val["Degree"] - 1]* 100:.2f}</p>',
            unsafe_allow_html=True,
        )

        # Create a line chart for poly kernel
        fig = px.line(
            df_poly, x="degree", y=["acc", "f1"], markers=True, title="Poly Kernel"
        )

    # Highlight the selected value on the graph
    selected_value = val.get("C", val.get("Degree", 1))
    fig.add_shape(
        type="line",
        x0=selected_value,
        x1=selected_value,
        y0=min(df_rbf["acc"].min(), df_rbf["f1"].min())
        if kernel_selector == "Radial Basis Function"
        else min(df_poly["acc"].min(), df_poly["f1"].min()),
        y1=max(df_rbf["acc"].max(), df_rbf["f1"].max())
        if kernel_selector == "Radial Basis Function"
        else max(df_poly["acc"].max(), df_poly["f1"].max()),
        line=dict(color="red", width=2, dash="dash"),
    )

    # Customize the legend labels
    fig.update_layout(legend=dict(title_text="Score"))

    # Display the graph in Streamlit
    st.plotly_chart(fig)


if section == "Detect handwrite digit":
    model = load_model()

    canvas = st_canvas(
        stroke_width=70,
        stroke_color="white",  # Đặt màu nét vẽ thành màu trắng (255, 255, 255 là màu trắng)
        background_color="black",  # Đặt màu nền thành màu đen (0, 0, 0 là màu đen)
        drawing_mode="freedraw",
        key="drawing_canvas",
        height=500,
        width=500,
    )
    if st.button("Detect handwrite digit"):
        if canvas.image_data is not None:
            try:
                # Convert the canvas drawing to a NumPy array
                img_data = np.array(canvas.image_data, dtype=np.uint8)
                # Resize the image to 28x28 pixels
                img_data = img_data[:, :, :3]
                img_data = cv2.resize(img_data, (28, 28))
                # Convert to grayscale
                img_data_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                flattened_image = img_data_gray.flatten()
                # Reshape the flattened image into a 2D array with a single sample
                flattened_image = flattened_image.reshape(1, -1)
                # Use the SVM model to make predictions
                prediction = model.predict(flattened_image)

                # Display the processed image
                st.image(img_data_gray, caption="Processed Image")

                # Display the prediction result
                st.write(f"Predicted Figure: {prediction[0]}")
            except Exception as e:
                st.write(f"Error processing image: {str(e)}")
        else:
            st.write("Vui lòng vẽ trước khi nhấn 'Detect handwrite digit'.")

if section == "Upload Image":
    with open("results_svm/specify_rbf/svm_model_rbf_C4.pickle", "rb") as model_file:
        model = pickle.load(model_file)

    # Create a function to preprocess and predict the digit from an image
    def predict_digit(image):
        image = np.array(image)
        # Preprocess the image (resize, grayscale, flatten)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image.flatten()

        # Make the prediction using the SVM model
        prediction = model.predict([image])

        return prediction[0]

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            prediction = predict_digit(image)
            st.write(f"Predicted Digit: {prediction}")
