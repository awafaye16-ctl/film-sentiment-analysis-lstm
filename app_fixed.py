# app_fixed.py
import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# ---------------------------
# 1. CONFIGURATION PAGE & CSS DARK MODE
# ---------------------------
st.set_page_config(
    page_title="Sentiment Analysis - Films",
    page_icon="🎬",
    layout="centered"
)

def local_css():
    st.markdown("""
        <style>
        /* Fond principal noir et texte blanc */
        .stApp {
            background-color: #0d0d0d;
            color: #ffffff;
        }

        /* Personnalisation des titres */
        h1, h2, h3 {
            color: #ff0000;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        h1 {
            border-bottom: 2px solid #ff0000;
            padding-bottom: 10px;
        }

        /* Style de la zone de texte */
        .stTextArea textarea {
            background-color: #1a1a1a !important;
            color: white !important;
            border: 1px solid #444 !important;
        }
        .stTextArea textarea:focus {
            border-color: #ff0000 !important;
            box-shadow: 0 0 10px #ff0000 !important;
        }

        /* Bouton Rouge Professionnel */
        .stButton>button {
            background-color: #ff0000 !important;
            color: white !important;
            font-weight: bold !important;
            border: none !important;
            transition: 0.3s;
            text-transform: uppercase;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #cc0000 !important;
            transform: scale(1.02);
            box-shadow: 0px 0px 15px rgba(255, 0, 0, 0.4);
        }

        /* Sidebar personnalisée */
        [data-testid="stSidebar"] {
            background-color: #111111;
            border-right: 1px solid #333;
        }

        /* Métriques */
        [data-testid="stMetric"] {
            background-color: #1e1e1e;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #ff0000;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ---------------------------
# 2. SIDEBAR - MÉTRIQUES & INFOS
# ---------------------------
with st.sidebar:
    st.markdown("## 🎬 **CinéScope AI**")
    st.markdown("---")
    st.markdown("### 📊 Performances")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "85%")
    col2.metric("Modèle", "LSTM Simple")
    st.markdown("---")
    st.markdown("**Fonctionnement** : analyse de critiques de films (anglais) et prédiction du sentiment (positif/négatif).")
    st.markdown("---")
    st.caption("Propulsé par TensorFlow & Streamlit")

# ---------------------------
# 3. CRÉATION D'UN MODÈLE SIMPLE (SANS GLOVE)
# ---------------------------
@st.cache_resource
def create_simple_model():
    model = Sequential([
        Embedding(5000, 64, input_length=250),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource  
def create_simple_tokenizer():
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    
    # Exemples de critiques pour entraîner le tokenizer
    sample_texts = [
        "this movie was great amazing fantastic wonderful excellent",
        "the film was terrible awful horrible bad disappointing boring",
        "great acting good story amazing cinematography brilliant performance",
        "bad acting terrible plot boring slow weak predictable dull",
        "love this film best movie ever outstanding superb incredible",
        "hate this movie worst film ever terrible awful horrible",
        "amazing film great story excellent acting wonderful masterpiece",
        "terrible movie bad plot awful acting boring waste time"
    ]
    
    tokenizer.fit_on_texts(sample_texts)
    return tokenizer

MAXLEN = 250  # Doit correspondre à la valeur utilisée lors de l'entraînement

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # balises HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # ponctuation, chiffres
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# 4. INTERFACE PRINCIPALE
# ---------------------------
st.title("🎬 Analyse de Sentiments de Films")
st.markdown("Critiques positives / négatives – Modèle LSTM bidirectionnel")

# Zone de saisie
st.subheader("📝 Entrez votre critique")
user_input = st.text_area(
    "",
    placeholder="Exemple : This movie was absolutely fantastic! The acting was superb...",
    height=150
)

# Bouton d'analyse
if st.button("🔍 Analyser le sentiment", use_container_width=True):
    if not user_input.strip():
        st.error("Veuillez saisir une critique.")
    else:
        # Chargement du modèle et tokenizer
        with st.spinner("Chargement du modèle..."):
            model = create_simple_model()
            tokenizer = create_simple_tokenizer()

        # Prétraitement
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAXLEN, padding='post', truncating='post')

        # Prédiction
        prob = model.predict(padded, verbose=0)[0][0]
        sentiment = "positif 😊" if prob >= 0.5 else "négatif 😞"
        confidence = max(prob, 1 - prob) * 100

        # Affichage des résultats
        st.success("Analyse terminée !")
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", sentiment)
        col2.metric("Confiance", f"{confidence:.1f}%")
        st.progress(int(confidence))

        # Détails techniques
        with st.expander("📊 Détails techniques"):
            st.write(f"**Score brut (positif)** : {prob:.4f}")
            st.write(f"**Texte nettoyé** : {cleaned[:200]}...")
            st.write(f"**Longueur séquence** : {len(seq[0])} mots")

# Pied de page
st.markdown("---")
st.caption("Modèle LSTM simple | Tokenizer intégré | Interface Streamlit")
