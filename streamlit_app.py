import streamlit as st
import pandas as pd

from function_predict import predict

# Titre de l'application
st.title('Analyse de Sentiment des Tweets')

# Texte d'instructions
st.write('Entrez un tweet dans la zone de texte ci-dessus et cliquez sur "Analyser le Sentiment" pour obtenir une prédiction de sentiment !')

# Zone de texte pour saisir le tweet
tweet_input = st.text_area('Saisissez votre tweet ici :')

# Bouton de prédiction
if st.button('Analyser le Sentiment'):
    if tweet_input:
        sentiment = predict(tweet_input)
        st.write(f"Sentiment : {sentiment}")
    else:
        st.warning('Veuillez saisir un tweet pour l\'analyse.')

# Titre de l'application
st.title("Data Analyse")


# Titre de l'application
st.title("Affichage du Wordcloud du modèle Keras")

# Chargez et affichez l'image PNG
wordwloud = "wordcloud1.png"  # Remplacez "mon_graphique.png" par le chemin vers votre image PNG
st.image(wordwloud, caption="Wordcloud du dataset", use_column_width=True)

# Titre de l'application
st.title("Affichage des Wordclouds du modèles ROBERTa enfonction de la classe")

# Créer deux colonnes
col1, col2, col3 = st.columns(3)

# Charger et afficher la première image PNG (Wordcloud Neutre)
wordcloud_neutre = "wordcloud2.png"  # Remplacez par le chemin vers votre image PNG Neutre
col1.image(wordcloud_neutre, caption="Wordcloud des mots Positif", use_column_width=True)

# Charger et afficher la première image PNG (Wordcloud Neutre)
wordcloud_neutre = "wordcloud3.png"  # Remplacez par le chemin vers votre image PNG Neutre
col2.image(wordcloud_neutre, caption="Wordcloud des mots Neutre", use_column_width=True)

# Charger et afficher la deuxième image PNG (Wordcloud Négatif)
wordcloud_negatif = "wordcloud4.png"  # Remplacez par le chemin vers votre image PNG Négatif
col3.image(wordcloud_negatif, caption="Wordcloud des mots Négatif", use_column_width=True)