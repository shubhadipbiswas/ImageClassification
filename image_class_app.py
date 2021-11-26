import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Vehicle Classifier: Motor Cycle(MC) or Scooter')

st.markdown("Welcome to this simple web application that classifies Vehicles ")
st.write("""
        
This app predicts the **Vehicles: Motor Cycle or Scooter!
""")


def main():
    
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', width=100)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write("""
             ### Prediction
                          """)
                st.write(predictions)
                st.pyplot(fig,width=100)


def predict(image):
    classifier_model = "model_image.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'Motor Cycle',
          'Scooter']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Motor Cycle': 0,
          'Scooter': 0}

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result









    

if __name__ == "__main__":
    main()
