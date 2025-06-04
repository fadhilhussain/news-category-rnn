import streamlit as st
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle



# Load tokenizer and label encoder
with open('models/tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)


with open('models/label.pkl','rb') as file:
    labels_categ = pickle.load(file)

# Load trained model
model = load_model('models/news_lstm.h5')


## data cleaning
def clean_data(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]',' ',text)
    return text

## prediction
def predict_news_categ(model,tokenizer,text,sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list],maxlen=sequence_length,padding='post',truncating='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_categ = np.argmax(predicted,axis=1)[0]
    predicted_label = labels_categ.inverse_transform([predicted_categ])[0]
    return predicted_label

# Streamlit UI
st.set_page_config(page_title="📰 News Category Classifier", layout="centered")

st.markdown("""
    <div style="text-align:center">
        <h2 style="color:#4A90E2;">🗞️ News Category Prediction</h2>
        <p>Enter a news headline or short paragraph below to predict its category.</p>
    </div>
    """, unsafe_allow_html=True)

user_input = st.text_area("📝 Enter News Text", height=150, placeholder="e.g. Apple launches new product in tech event...")

if st.button("🔍 Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        text = clean_data(user_input)
        print(text)
        max_length_of_sequence = model.input_shape[1]
        prediction = predict_news_categ(model,tokenizer,text,max_length_of_sequence)

        st.success(f"✅ **Predicted Category:** {prediction}")
