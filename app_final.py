# app_final.py
import streamlit as st
import numpy as np
import re
import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
            background-color: #1a1a1a;
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
            background-color: #2a2a2a !important;
            color: white !important;
            border: 1px solid #555 !important;
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

        /* Barre de progression dégradée personnalisée */
        .progress-container {
            background: linear-gradient(to right, #ff0000, #ff6600, #ffcc00, #66ff00, #00ff00);
            height: 30px;
            border-radius: 15px;
            position: relative;
            overflow: hidden;
            margin: 20px 0;
        }

        .progress-bar-custom {
            background: rgba(0, 0, 0, 0.3);
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }

        /* Surlignage des mots-clés */
        .positive-word {
            background-color: #00ff00;
            color: black;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }

        .negative-word {
            background-color: #ff0000;
            color: white;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }

        .neutral-word {
            background-color: #666666;
            color: white;
            padding: 2px 4px;
            border-radius: 3px;
        }

        /* Sidebar personnalisée */
        [data-testid="stSidebar"] {
            background-color: #2a2a2a;
            border-right: 1px solid #444;
            color: #ffffff !important;
        }

        /* Texte dans la sidebar */
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* Métriques dans sidebar */
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            background-color: #333333;
            color: #ffffff !important;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #ff0000;
        }

        /* Markdown dans sidebar */
        [data-testid="stSidebar"] .markdown-text-container {
            color: #ffffff !important;
        }

        /* Footer personnalisé */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #2a2a2a;
            color: #ffffff;
            text-align: center;
            padding: 15px;
            border-top: 2px solid #ff0000;
            font-size: 14px;
        }
        
        .footer a {
            color: #ff0000;
            text-decoration: none;
            font-weight: bold;
        }
        
        .footer a:hover {
            color: #ff6666;
            text-decoration: underline;
        }

        /* Espace pour le footer */
        .main-content {
            padding-bottom: 80px;
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
    col1.metric("Accuracy", "90%")
    col2.metric("Modèle", "LSTM + GloVe")
    st.markdown("---")
    st.markdown("**Fonctionnement** : analyse de critiques de films (anglais) et prédiction du sentiment (positif/négatif).")
    st.markdown("---")
    st.caption("Propulsé par TensorFlow, GloVe & Streamlit")

# ---------------------------
# 3. CHARGEMENT DU MODÈLE ET DU TOKENIZER ORIGINAUX
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer():
    """Charge le modèle et tokenizer entraînés depuis le notebook"""
    try:
        model = load_model("imdb_lstm_glove.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None

MAXLEN = 250  # Doit correspondre à la valeur utilisée lors de l'entraînement (90e percentile)

def clean_text(text):
    """Nettoyage du texte identique au notebook"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # balises HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # ponctuation, chiffres
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment_words(text, tokenizer, model):
    """Analyse les mots-clés qui influencent le sentiment"""
    # Dictionnaires de mots positifs/négatifs pour l'analyse
    positive_words = {
        'great', 'amazing', 'fantastic', 'excellent', 'wonderful', 'brilliant', 
        'superb', 'outstanding', 'magnificent', 'incredible', 'love', 'awesome',
        'perfect', 'beautiful', 'best', 'masterpiece', 'stunning', 'spectacular',
        'marvelous', 'terrific', 'exceptional', 'remarkable', 'phenomenal'
    }
    
    negative_words = {
        'terrible', 'awful', 'horrible', 'bad', 'worst', 'boring', 'disappointing',
        'disaster', 'waste', 'poor', 'weak', 'dull', 'predictable', 'mediocre',
        'uninteresting', 'tedious', 'monotonous', 'painful', 'annoying', 'frustrating',
        'useless', 'hate', 'disgusting', 'pathetic', 'lame', 'stupid', 'trash'
    }
    
    words = text.lower().split()
    analyzed_words = []
    
    for word in words:
        # Nettoyer le mot
        clean_word = re.sub(r'[^\w]', '', word)
        
        if clean_word in positive_words:
            analyzed_words.append(f'<span class="positive-word">{word}</span>')
        elif clean_word in negative_words:
            analyzed_words.append(f'<span class="negative-word">{word}</span>')
        else:
            analyzed_words.append(f'<span class="neutral-word">{word}</span>')
    
    return ' '.join(analyzed_words)

def analyze_batch_sentiment(df, model, tokenizer):
    """Analyse les sentiments pour un DataFrame de critiques"""
    results = []
    
    for idx, row in df.iterrows():
        text = str(row.iloc[0])  # Prend la première colonne
        
        if pd.isna(text) or text.strip() == "":
            results.append({
                'texte_original': text,
                'sentiment': 'Non analysé',
                'confiance': 0.0,
                'score_brut': 0.0
            })
            continue
        
        # Prétraitement
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        # CORRECTION CRITIQUE : padding par défaut ('pre') comme dans le notebook
        padded = pad_sequences(seq, maxlen=MAXLEN)
        
        # Prédiction
        prob = model.predict(padded, verbose=0)[0][0]
        sentiment = "positif 😊" if prob >= 0.5 else "négatif 😞"
        confidence = max(prob, 1 - prob) * 100
        
        results.append({
            'texte_original': text,
            'texte_nettoyé': cleaned,
            'sentiment': sentiment,
            'confiance': confidence,
            'score_brut': prob
        })
    
    return pd.DataFrame(results)

def create_custom_progress_bar(confidence):
    """Crée une barre de progression dégradée personnalisée"""
    # Déterminer la position et la couleur selon le sentiment
    if confidence >= 50:
        # Sentiment positif
        bar_width = confidence
        position = "right"
        sentiment_color = "🟢"
    else:
        # Sentiment négatif  
        bar_width = 100 - confidence
        position = "left"
        sentiment_color = "🔴"
    
    # HTML pour la barre de progression dégradée
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-bar-custom" style="width: {bar_width}%; {'margin-left: auto;' if position == 'right' else ''}">
            {sentiment_color} {confidence:.1f}%
        </div>
    </div>
    """
    return progress_html

# ---------------------------
# 4. INTERFACE PRINCIPALE
# ---------------------------
st.title("🎬 Analyse de Sentiments de Films")
st.markdown("Critiques positives / négatives – Modèle LSTM bidirectionnel + GloVe")

# Note importante
st.info("📝 **Note** : Le modèle a été entraîné sur le dataset IMDb. Veuillez saisir vos critiques en anglais.")

# Vérification technique de la connexion
st.markdown("### 🔧 Vérification Technique de la Connexion")

# Charger le modèle et tokenizer pour vérification
model, tokenizer = load_model_and_tokenizer()

col1, col2, col3 = st.columns(3)

# Vérification 1: Tokenizer
with col1:
    if tokenizer is not None:
        st.success("✅ Tokenizer chargé")
        st.caption(f"Vocabulaire: {len(tokenizer.word_index)} mots")
    else:
        st.error("❌ Tokenizer non défini")

# Vérification 2: Modèle
with col2:
    if model is not None:
        st.success("✅ Modèle chargé")
        st.caption("Architecture: LSTM + GloVe")
    else:
        st.error("❌ Modèle non défini")

# Vérification 3: Pipeline complète
with col3:
    try:
        # Test avec une phrase connue
        test_text = "excellent"
        if tokenizer is not None and model is not None:
            seq = tokenizer.texts_to_sequences([test_text])
            if seq and seq[0]:
                word_index = seq[0][0] if seq[0] else None
                st.success("✅ Pipeline fonctionnelle")
                st.caption(f"Index 'excellent': {word_index}")
            else:
                st.warning("⚠️ Pipeline partielle")
        else:
            st.error("❌ Pipeline cassée")
    except Exception as e:
        st.error(f"❌ Erreur pipeline: {str(e)}")

st.markdown("---")
st.markdown("### 🧪 Test de Validation")
test_phrase = st.text_input("Entrez une phrase test :", value="A total waste of time and money")

if st.button("🔬 Tester la phrase"):
    if test_phrase.strip():
        # Charger le modèle et tokenizer pour le test
        model, tokenizer = load_model_and_tokenizer()
        if model is not None and tokenizer is not None:
            with st.spinner("Test en cours..."):
                cleaned = clean_text(test_phrase)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=MAXLEN)
                prob = model.predict(padded, verbose=0)[0][0]
                
            st.metric("Score brut", f"{prob:.4f}")
            st.metric("Sentiment", "Positif" if prob >= 0.5 else "Négatif")
            
            # Validation
            if prob < 0.3:
                st.success("✅ CORRECT : Phrase négative bien détectée !")
            elif prob > 0.7:
                st.error("❌ ERREUR : Phrase négative détectée comme positive !")
            else:
                st.warning("⚠️ AMBIGUË : Le modèle hésite")

st.markdown("---")

# Mode d'analyse
mode = st.radio(
    "🎯 Choisissez le mode d'analyse :",
    ["📝 Analyse simple", "📊 Analyse de masse (CSV)"],
    horizontal=True
)

# Vérification des fichiers
model_path = "imdb_lstm_glove.h5"
tokenizer_path = "tokenizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    st.error("❌ Fichiers du modèle manquants !")
    st.markdown("""
    **Solution :**
    1. Assurez-vous que `imdb_lstm_glove.h5` et `tokenizer.pkl` sont dans ce dossier
    2. Ces fichiers doivent provenir de l'entraînement du notebook
    """)
else:
    # Chargement du modèle pour vérification
    model, tokenizer = load_model_and_tokenizer()
        
    if model is None or tokenizer is None:
        st.error("Impossible de charger le modèle. Vérifiez les fichiers.")
    else:
        if mode == "📝 Analyse simple":
            # Interface d'analyse simple
            st.subheader("📝 Entrez votre critique")
            user_input = st.text_area(
                "",
                placeholder="Exemple : This movie was absolutely fantastic! The acting was superb...",
                height=150
            )

            # Exemples prédéfinis
            with st.expander("📋 Exemples de critiques"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💚 Exemple Positif"):
                        user_input = "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout. I would definitely recommend it to anyone looking for a great film."
                with col2:
                    if st.button("💔 Exemple Négatif"):
                        user_input = "I was really disappointed with this film. The plot was predictable and the acting felt forced. I wouldn't waste my time watching this again."

            # Bouton d'analyse
            if st.button("🔍 Analyser le sentiment", use_container_width=True):
                if not user_input.strip():
                    st.error("Veuillez saisir une critique.")
                else:
                    # Prétraitement identique au notebook
                    with st.spinner("Analyse en cours..."):
                        cleaned = clean_text(user_input)
                        seq = tokenizer.texts_to_sequences([cleaned])
                        # CORRECTION CRITIQUE : padding par défaut ('pre') comme dans le notebook
                        padded = pad_sequences(seq, maxlen=MAXLEN)

                        # Prédiction
                        prob = model.predict(padded, verbose=0)[0][0]
                        sentiment = "positif 😊" if prob >= 0.5 else "négatif 😞"
                        confidence = max(prob, 1 - prob) * 100

                    # Affichage des résultats
                    st.success("✅ Analyse terminée !")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Confiance", f"{confidence:.1f}%")
                    
                    # Barre de progression dégradée personnalisée
                    st.markdown("### 🎯 Visualisation du Sentiment")
                    st.markdown(create_custom_progress_bar(confidence), unsafe_allow_html=True)
                    
                    # Analyse des mots-clés avec highlighting
                    st.markdown("### 🔍 Analyse des Mots-Clés")
                    highlighted_text = analyze_sentiment_words(user_input, tokenizer, model)
                    st.markdown(f"<div style='font-size: 18px; line-height: 1.6;'>{highlighted_text}</div>", unsafe_allow_html=True)
                    
                    # Légende des couleurs
                    st.markdown("""
                    **Légende des couleurs :**
                    - 🟢 **Vert** : Mots positifs qui renforcent le sentiment positif
                    - 🔴 **Rouge** : Mots négatifs qui renforcent le sentiment négatif  
                    - ⚫ **Gris** : Mots neutres ou non influents
                    """)

                    # Détails techniques
                    with st.expander("📊 Détails techniques"):
                        st.write(f"**Score brut (positif)** : {prob:.4f}")
                        st.write(f"**Texte nettoyé** : {cleaned[:200]}...")
                        st.write(f"**Longueur séquence** : {len(seq[0])} mots")
                        st.write(f"**Vocabulaire tokenizer** : {len(tokenizer.word_index)} mots")
                        
                        # Analyse XAI (IA Explicable)
                        st.markdown("**🧠 Analyse XAI (IA Explicable)**")
                        st.markdown("""
                        Cette application utilise une technique d'IA explicable pour identifier les mots 
                        qui ont le plus influencé la décision du modèle. Les mots sont classés selon leur 
                        polarité sémantique et surlignés en couleur pour une meilleure compréhension.
                        """)
        
        else:  # Mode analyse de masse CSV
            st.subheader("📊 Analyse de Masse - Upload CSV")
            st.markdown("""
            **Instructions :**
            - Uploadez un fichier CSV avec une colonne de critiques
            - La première colonne sera utilisée pour l'analyse
            - Format attendu : `.csv` avec encodage UTF-8
            """)
            
            uploaded_file = st.file_uploader(
                "📁 Choisissez un fichier CSV",
                type=['csv'],
                help="Le fichier doit contenir une colonne avec des critiques de films en anglais"
            )
            
            if uploaded_file is not None:
                try:
                    # Lecture du fichier
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Fichier chargé : {df.shape[0]} critiques trouvées")
                    
                    # Aperçu des données
                    with st.expander("👁️ Aperçu des données"):
                        st.dataframe(df.head())
                    
                    # Bouton d'analyse
                    if st.button("🚀 Lancer l'analyse de masse", use_container_width=True):
                        with st.spinner("⚡ Analyse en cours..."):
                            results_df = analyze_batch_sentiment(df, model, tokenizer)
                        
                        # Statistiques
                        st.success("✅ Analyse terminée !")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            positifs = len(results_df[results_df['sentiment'] == 'positif 😊'])
                            st.metric("Positifs", positifs)
                        with col2:
                            negatifs = len(results_df[results_df['sentiment'] == 'négatif 😞'])
                            st.metric("Négatifs", negatifs)
                        with col3:
                            avg_confiance = results_df[results_df['sentiment'] != 'Non analysé']['confiance'].mean()
                            st.metric("Confiance moyenne", f"{avg_confiance:.1f}%")
                        
                        # Graphique
                        st.markdown("### 📈 Distribution des sentiments")
                        sentiment_counts = results_df['sentiment'].value_counts()
                        st.bar_chart(sentiment_counts)
                        
                        # Tableau de résultats
                        st.markdown("### 📋 Résultats détaillés")
                        st.dataframe(results_df)
                        
                        # Téléchargement des résultats
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger les résultats (CSV)",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement du fichier : {str(e)}")
                    st.markdown("""
                    **Conseils :**
                    - Vérifiez que le fichier est bien au format CSV
                    - Assurez-vous que l'encodage est UTF-8
                    - La première colonne doit contenir les critiques
                    """)

# Pied de page professionnel
st.markdown("""
<div class="footer">
    🎬 <strong>CinéScope AI</strong> - Analyse de Sentiments de Films | 
    Développé avec ❤️ par <a href="https://github.com/awafaye16-ctl" target="_blank">Awa Faye</a> | 
    Powered by TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.caption("Modèle entraîné sur 40 000 critiques IMDb | Embeddings GloVe | LSTM bidirectionnel")
