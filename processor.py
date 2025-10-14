import os
import json
import pickle
import random
import logging
import traceback
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)

# --- Ensure NLTK resources are available ---
for pkg in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

# --- Initialize paths ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'chatbot_model.h5')
INTENTS_PATH = os.path.join(BASE_DIR, 'intents.json')
WORDS_PATH = os.path.join(BASE_DIR, 'words.pkl')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.pkl')

lemmatizer = WordNetLemmatizer()

# --- Load model and data ---
try:
    model = load_model(MODEL_PATH)
    logging.info("✅ Chatbot model loaded successfully.")
except Exception:
    logging.error("❌ Failed to load model:\n" + traceback.format_exc())
    model = None

try:
    with open(WORDS_PATH, 'rb') as f:
        words = pickle.load(f)
    with open(CLASSES_PATH, 'rb') as f:
        classes = pickle.load(f)
    with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
        intents = json.load(f)
    logging.info("✅ Words, classes, and intents loaded.")
except Exception:
    logging.error("❌ Failed to load words/classes/intents:\n" + traceback.format_exc())
    words, classes, intents = [], [], {}

# --- Helper functions ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    if model is None:
        raise RuntimeError("Model not loaded.")
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    try:
        tag = intents_list[0]['intent']
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except Exception:
        pass
    return "I'm not sure I understand you."

# --- Main chatbot function ---
def chatbot_response(user_input):
    try:
        intents_list = predict_class(user_input)
        result = get_response(intents_list)
        return result
    except Exception:
        logging.error("Error in chatbot_response:\n" + traceback.format_exc())
        return "Sorry, something went wrong!"
