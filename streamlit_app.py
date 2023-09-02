import streamlit as st
import pandas as pd

from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras

import pickle
from keras import backend as K

import matplotlib.pyplot as plt


#local variables
var_model_path = 'D:/anaconda3/envs/env1/notebooks/OP Notebooks/p10/Github/mon_model.h5'
var_word_embeding_path = 'D:/anaconda3/envs/env1/notebooks/OP Notebooks/p10/Github/tokenizer_GLOVE_LSTM_traite.pkl'
var_dataset_summary = 'D:/anaconda3\envs\env1/notebooks\OP Notebooks\p10\Github/summary.csv'
'''

#variables
var_model_path = 'mon_model.h5'
var_word_embeding_path = 'tokenizer_GLOVE_LSTM_traite.pkl'
var_dataset_summary = 'summary.csv'
var = '''
# Fonctions pour les métriques personnalisées
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


def decode_sentiment(score):
    if float(score) < 0.5:
        label = "NEGATIVE"

        return label
    else:
        label = "POSITIVE"

        return label

def predict(text):
    model = keras.models.load_model(var_model_path, compile = False)

    model.compile(loss=keras.losses.binary_crossentropy,
                metrics=['accuracy', recall, precision, f1_score]
                )

    tokenizer = pickle.load(open(var_word_embeding_path, "rb"))

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=30)

    score = model.predict([x_test])[0]

    label = decode_sentiment(score)

    return label


# Titre de l'application
st.title('Analyse de Sentiment des Tweets')

# Texte d'instructions
st.write('Entrez un tweet dans la zone de texte ci-dessus et cliquez sur "Analyser le Sentiment" pour obtenir une prédiction de sentiment.')

# Note de fin
st.write('Ce modèle d\'analyse de sentiment a été créé à des fins de démonstration.')

# Zone de texte pour saisir le tweet
tweet_input = st.text_area('Saisissez votre tweet ici :')

# Bouton de prédiction
if st.button('Analyser le Sentiment'):
    if tweet_input:
        sentiment = predict(tweet_input)
        st.write(f"Sentiment : {sentiment}")
    else:
        st.warning('Veuillez saisir un tweet pour l\'analyse.')


# Charger votre DataFrame (remplacez ceci par vos données)
df = pd.read_csv(var_dataset_summary)

# Titre de l'application
st.title('Analyse Statistique')

# Analyse statistique avec describe()
st.write("Analyse statistique des données :")
st.write(df.describe())


# Titre de l'application
st.title("Affichage de la distribution de classe")

# Chargez et affichez l'image PNG
image = "class_distribution.png"  # Remplacez "mon_graphique.png" par le chemin vers votre image PNG
st.image(image, caption="Distribution de la classe target", use_column_width=True)

# Titre de l'application
st.title("Affichage de la distribution de classe")

# Chargez et affichez l'image PNG
wordwloud = "wordcloud.png"  # Remplacez "mon_graphique.png" par le chemin vers votre image PNG
st.image(wordwloud, caption="Wordcloud du dataset", use_column_width=True)