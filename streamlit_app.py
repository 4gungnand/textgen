import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pickle

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background: url("https://wallpapercrafter.com/desktop1/539525-video-games-Portal-game-black-background-copy.jpg");
background-size: cover;
opacity: 1;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


@st.cache_resource()
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('model.h5')
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def text_generator(input_text, next_words=20, model=model, tokenizer=tokenizer):
    seed_text = input_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=20, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted, axis=-1).item()
        output_word = tokenizer.index_word[predicted]
        seed_text += " " + output_word
    return seed_text


st.title("Game Dialogue Generator")
st.write("This app generates dialogue based on the input text. The model was trained on dialogue from the game 'Portal' by Valve.")
input_text = st.text_input("Enter your seed text:")
next_words = st.slider("How many words do you want to generate?", 1, 50)

if st.button("Generate"):
    st.subheader("Generated Dialogue:")
    with st.spinner(text="This may take a moment..."):
        output_text = text_generator(input_text, next_words, model, tokenizer)
    st.text_area("", output_text, height=120, max_chars=None, key=None)

    
    
