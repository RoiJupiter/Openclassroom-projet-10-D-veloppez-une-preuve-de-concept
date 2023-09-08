import numpy as np

from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel  
import tensorflow as tf

MODEL_PATH = "D:/anaconda3/envs/env1/notebooks/OP Notebooks/p10/Github/model_roberta.h5"
MODEL_PATH = "model_roberta.h5"
MAX_LEN = 128

# Créez un dictionnaire de correspondance pour les couches personnalisées
custom_objects = {"TFRobertaModel": TFRobertaModel}
# Chargez le modèle en spécifiant les custom_objects
model = tf.keras.models.load_model(MODEL_PATH, custom_objects = custom_objects)
tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")

def predict(text):
    val_input_ids, val_attention_masks = tokenize_roberta([text])
    predictions = model.predict([val_input_ids, val_attention_masks])
    y_pred_roberta =  np.zeros_like(predictions)
    y_pred_roberta[np.arange(len(y_pred_roberta)), predictions.argmax(1)] = 1
    # Extraire la classe prédite en accédant à la première liste interne et en utilisant argmax
    classe_predite = np.argmax(y_pred_roberta[0])

    # Créer un dictionnaire de correspondance entre les indices de classe et les libellés de classe
    classes = {0: "Négatif", 1: "Neutre", 2: "Positif"}

    '''# Afficher le libellé de classe en fonction de la classe prédite
    if classe_predite in classes:
        print(f"La classe prédite est '{classes[classe_predite]}'.")
    else:
        print("La classe prédite n'est pas reconnue.")'''
    
    return classes[classe_predite]

def tokenize_roberta(data,max_len=MAX_LEN) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer_roberta.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


textn = "You were supposed to kill them not join them"
textp = "You are beautiful as the day as I lost you"
print(predict(textp))
