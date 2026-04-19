# Film Sentiment Analysis LSTM

## Description

Application web d'analyse de sentiments de critiques de films utilisant un réseau de neurones LSTM bidirectionnel avec embeddings GloVe.

## Fonctionnalités

- Analyse de sentiment en temps réel
- Interface utilisateur intuitive
- Modèle LSTM avec 88% d'accuracy
- Visualisation des résultats et niveau de confiance

## Technologies

- TensorFlow / Keras
- Streamlit
- Python
- NumPy
- Pandas
- GloVe Embeddings

## Installation

1. Cloner le dépôt
2. Créer un environnement virtuel
3. Installer les dépendances
4. Lancer l'application

## Utilisation

Lancer l'application avec :
```bash
streamlit run app.py
```

## Architecture

- Embedding GloVe pré-entraîné
- LSTM bidirectionnel (64 unités)
- Couches de Dropout pour la régularisation
- Sortie binaire (positif/négatif)

## Performance

- Accuracy : 88%
- Modèle entraîné sur 40 000 critiques IMDb
- Temps de réponse : < 1 seconde
