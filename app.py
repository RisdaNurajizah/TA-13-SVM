import streamlit as st
import pickle
import re

# Set Judul Aplikasi
st.set_page_config(page_title="Analisis Sentimen PPKM", page_icon="📊")

# Load model dan vectorizer
@st.cache_resource
def load_assets():
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_assets()

# Fungsi Preprocessing sederhana
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-z\s]', '', text)
    return text

# Antarmuka Pengguna
st.title("📱 Sentiment Analysis App - TA 13")
st.write("Aplikasi ini memprediksi apakah ulasan/tweet bermakna **Positif** atau **Negatif** menggunakan SVM.")

user_input = st.text_area("Masukkan teks ulasan di sini:", placeholder="Contoh: PPKM ini sangat membantu menekan angka covid...")

if st.button("Prediksi Sentimen"):
    if user_input:
        # 1. Preprocessing
        cleaned_text = clean_text(user_input)
        
        # 2. Transformasi ke TF-IDF
        vectorized_text = tfidf.transform([cleaned_text])
        
        # 3. Prediksi
        prediction = model.predict(vectorized_text)
        
        # 4. Tampilkan Hasil
        if prediction[0] == 1:
            st.success("Hasil: **SENTIMEN POSITIF** 😊")
        else:
            st.error("Hasil: **SENTIMEN NEGATIF** 😡")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")