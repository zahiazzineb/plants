import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 

st.header('Zahia Zineb Online Learning Formation \n Plants diseases Classification Model')
model = load_model('cnn_model.h5')
data_cat = ["Blight","Common_Rust","Gray_Leaf_Spot","Healthy"]
img_height = 256
img_width = 256
image =st.text_input('Enter Image name','Corn_Gray_Spot (1).jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))