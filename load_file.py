import streamlit as st
import cv2
import numpy as np
import pickle

# Load a pre-trained SVM model for digit classification (You should have a trained model)
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


# Create the Streamlit web app
def main():
    st.title("SVM Digit Recognition")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            prediction = predict_digit(image)
            st.write(f"Predicted Digit: {prediction}")


if __name__ == "__main__":
    main()
