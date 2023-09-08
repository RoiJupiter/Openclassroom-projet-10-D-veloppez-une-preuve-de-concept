import streamlit as st
import pandas as pd

from function_predict import predict

from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras

import pickle
from keras import backend as K

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

# Chargez et affichez l'image PNG
image = "class_distribution.png"  # Remplacez "mon_graphique.png" par le chemin vers votre image PNG
st.image(image, caption="Distribution de la classe target", use_column_width=True)

# Titre de l'application
st.title("Affichage de l'ancien Wordcloud")

# Chargez et affichez l'image PNG
wordwloud = "wordcloud.png"  # Remplacez "mon_graphique.png" par le chemin vers votre image PNG
st.image(wordwloud, caption="Wordcloud du dataset", use_column_width=True)

# Titre de l'application
st.title("Affichage du nouveau Wordcloud")

# Chargez et affichez l'image PNG
wordcloud2 = "wordcloud2.png"  # Remplacez "mon_graphique.png" par le chemin vers votre image PNG
st.image(wordcloud2, caption="Wordcloud du dataset", use_column_width=True)