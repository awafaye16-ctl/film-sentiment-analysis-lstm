import streamlit as st
import numpy as np
import re
import os
import requests
import zipfile
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Sentiments",
    page_icon="🎬",
    layout="centered"
)

# Titre
st.title("🎬 Analyse de Sentiments de Films")
st.write("Analysez le sentiment des critiques de films avec l'IA")

# Variables globales
MAX_WORDS = 5000
MAX_LEN = 100
EMBEDDING_DIM = 50

# Fonction de nettoyage simple
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenizer simple
@st.cache_resource
def get_tokenizer():
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    
    # Textes d'entraînement simples
    texts = [
        "this movie was great amazing fantastic wonderful",
        "the film was terrible awful horrible bad",
        "excellent acting good story great movie",
        "boring slow predictable disappointing",
        "love this film best movie ever",
        "hate this movie worst film ever",
        "brilliant outstanding superb incredible",
        "dull uninspired mediocre weak"
    ]
    
    tokenizer.fit_on_texts(texts)
    return tokenizer

# Modèle simple
@st.cache_resource
def create_simple_model():
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Fonction de prédiction simple
def predict_sentiment(text, tokenizer, model):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    
    prediction = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positif 😊" if prediction > 0.5 else "Négatif 😞"
    confidence = max(prediction, 1 - prediction) * 100
    
    return sentiment, confidence, prediction

# Interface principale
def main():
    tokenizer = get_tokenizer()
    model = create_simple_model()
    
    st.subheader("📝 Écrivez votre critique")
    
    # Zone de texte
    user_input = st.text_area(
        "Votre critique de film :",
        placeholder="Exemple: Ce film était vraiment génial avec des acteurs incroyables...",
        height=100
    )
    
    # Exemples
    if st.checkbox("Voir des exemples"):
        examples = {
            "Positif": "This movie was absolutely fantastic! Great acting and amazing story.",
            "Négatif": "Terrible film with bad acting and boring plot.",
            "Mixte": "The movie had some good moments but overall was just average."
        }
        
        for label, example in examples.items():
            if st.button(f"Exemple {label}"):
                user_input = example
    
    # Bouton d'analyse
    if st.button("🔍 Analyser le sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyse en cours..."):
                sentiment, confidence, raw_score = predict_sentiment(user_input, tokenizer, model)
            
            # Résultats
            st.success("Analyse terminée !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Confiance", f"{confidence:.1f}%")
            
            # Barre de progression
            st.progress(confidence / 100)
            
            # Détails
            with st.expander("📊 Détails"):
                st.write(f"**Score brut :** {raw_score:.4f}")
                st.write(f"**Texte nettoyé :** {clean_text(user_input)}")
        
        else:
            st.error("Veuillez écrire une critique avant d'analyser.")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 💡 Comment utiliser :
    1. Écrivez votre critique de film dans la zone de texte
    2. Cliquez sur "Analyser le sentiment"
    3. Consultez le résultat et le niveau de confiance
    
    ### 🤖 Technologie :
    - Réseau de neurones LSTM bidirectionnel
    - Analyse de sentiment binaire (Positif/Négatif)
    """)

if __name__ == "__main__":
    main()
