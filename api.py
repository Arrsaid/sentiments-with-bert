import os
import warnings
from transformers import logging

# --- Imports FastAPI & modèles ---
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from utils import preprocess_tweet_bert

import gdown
import zipfile

# --- Configuration de l'environnement ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


logging.set_verbosity_error()


# --- Téléchargement du modèle depuis Google Drive si non présent ---
model_path = "./bert_model"
model_zip = "bert_model.zip"
gdrive_file_id = "1lpG1xAKfvlT-FT8QDQrhYHFH02A9yg9T"

if not os.path.exists(model_path):
    print("Téléchargement du modèle BERT depuis Google Drive...")
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, model_zip, quiet=False)

    print("Décompression...")
    with zipfile.ZipFile(model_zip, "r") as zip_ref:
        zip_ref.extractall(model_path)
    os.remove(model_zip)

# --- Chargement du modèle et du tokenizer ---
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# --- Initialisation de l'application ---
app = FastAPI(title="API BERT - Analyse de sentiments")


# --- Schéma de requête ---
class Tweet(BaseModel):
    text: str


# Route de santé
@app.get("/")
def read_root():
    return {"message": "API de prédiction de sentiment avec BERT- en ligne"}


# --- Route de prédiction ---
@app.post("/predict")
def predict(tweet: Tweet):
    tweet = preprocess_tweet_bert(tweet.text)
    inputs = tokenizer(
        tweet, return_tensors="tf", truncation=True, padding=True, max_length=128
    )
    outputs = model(inputs)
    logits = outputs.logits.numpy()
    proba = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred_class = np.argmax(proba)

    sentiment = "positif" if pred_class == 1 else "négatif"
    confidence = round(float(proba[pred_class]), 4)

    return {"sentiment": sentiment, "confidence": confidence}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
