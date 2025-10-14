# processor.py
import os
import json
import pickle
import logging
import traceback
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Try to import TensorFlow Keras; if not available, we'll fallback to sklearn-like pickle model
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Ensure NLTK data
for pkg in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        logging.info(f"Downloading nltk package: {pkg}")
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(__file__)

# Files (names exactly as you uploaded)
MODEL_H5 = os.path.join(BASE_DIR, "chatbot_model.h5")
MODEL_PKL = os.path.join(BASE_DIR, "chatbot_model.pkl")
WORDS_PKL = os.path.join(BASE_DIR, "words.pkl")
CLASSES_PKL = os.path.join(BASE_DIR, "classes.pkl")
INTENTS_JSON = os.path.join(BASE_DIR, "job_intents.json")  # you uploaded job_intents.json

# Debug: print existence
logging.info(f"Checking files existence:")
for p in [MODEL_H5, MODEL_PKL, WORDS_PKL, CLASSES_PKL, INTENTS_JSON]:
    logging.info(f" - {os.path.basename(p)} exists? {os.path.exists(p)}")

# Load intents
intents = {}
if os.path.exists(INTENTS_JSON):
    try:
        with open(INTENTS_JSON, 'r', encoding='utf-8') as f:
            intents = json.load(f)
        logging.info("Loaded intents from job_intents.json")
    except Exception:
        logging.error("Failed to load intents json:\n" + traceback.format_exc())
else:
    logging.error("Intents json not found. Expected file: job_intents.json")

# Load words and classes
words = []
classes = []
try:
    if os.path.exists(WORDS_PKL):
        with open(WORDS_PKL, 'rb') as f:
            words = pickle.load(f)
        logging.info(f"Loaded words.pkl ({len(words)} tokens)")
    else:
        logging.error("words.pkl not found")
    if os.path.exists(CLASSES_PKL):
        with open(CLASSES_PKL, 'rb') as f:
            classes = pickle.load(f)
        logging.info(f"Loaded classes.pkl ({len(classes)} classes)")
    else:
        logging.error("classes.pkl not found")
except Exception:
    logging.error("Failed to load words/classes:\n" + traceback.format_exc())
    words, classes = [], []

# Load model: prefer .h5 (Keras) then .pkl
model = None
model_type = None
if os.path.exists(MODEL_H5) and TF_AVAILABLE:
    try:
        logging.info("Attempting to load Keras model (.h5)...")
        model = load_model(MODEL_H5)
        model_type = 'keras'
        logging.info("Keras model loaded.")
    except Exception:
        logging.error("Failed to load .h5 model:\n" + traceback.format_exc())
        model = None
elif os.path.exists(MODEL_PKL):
    try:
        logging.info("Attempting to load pickled model (.pkl)...")
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        model_type = 'pickle'
        logging.info("Pickle model loaded.")
    except Exception:
        logging.error("Failed to load .pkl model:\n" + traceback.format_exc())
        model = None
else:
    if os.path.exists(MODEL_H5) and not TF_AVAILABLE:
        logging.error("TensorFlow not available in runtime, cannot load .h5 model.")
    else:
        logging.error("No model file found (neither chatbot_model.h5 nor chatbot_model.pkl).")

# Helper funcs
def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    return tokens

def bow(sentence, words_list):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_list)
    for s in sentence_words:
        for i, w in enumerate(words_list):
            if w == s:
                bag[i] = 1
    return np.array(bag, dtype=float)

# Prediction wrapper that supports keras and sklearn-like pickled models
def predict_intents(sentence):
    if model is None:
        raise RuntimeError("Model not loaded (model is None)")

    if not words or not classes:
        raise RuntimeError("Vocabulary (words) or classes not loaded")

    x = bow(sentence, words)  # shape (len(words),)
    logging.info(f"Bag of words vector length: {len(x)}; expected vocab length: {len(words)}")

    # Keras model: expects array-like shape (1, n_features)
    if model_type == 'keras':
        try:
            preds = model.predict(np.array([x]), verbose=0)
            # pred may be shape (1, n_classes)
            preds = preds[0]  # first sample
            # build list of (class, prob)
            results = []
            for i, p in enumerate(preds):
                results.append((i, float(p)))
            results.sort(key=lambda item: item[1], reverse=True)
            # apply threshold
            ERROR_THRESHOLD = 0.25
            filtered = [ {"intent": classes[i], "probability": str(prob)} for i, prob in results if prob > ERROR_THRESHOLD ]
            if not filtered:
                # return top-1 even if under threshold
                idx, prob = results[0]
                filtered = [{"intent": classes[idx], "probability": str(prob)}]
            return filtered
        except Exception:
            logging.error("Keras model prediction failed:\n" + traceback.format_exc())
            raise

    # Pickle model (sklearn-like) - try predict_proba or predict
    elif model_type == 'pickle':
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([x])[0]
                results = sorted(list(enumerate(probs)), key=lambda i: i[1], reverse=True)
                ERROR_THRESHOLD = 0.25
                filtered = [{"intent": classes[i], "probability": str(p)} for i, p in results if p > ERROR_THRESHOLD]
                if not filtered:
                    i, p = results[0]
                    filtered = [{"intent": classes[i], "probability": str(p)}]
                return filtered
            else:
                pred = model.predict([x])[0]
                # If model.predict returns class index or class label
                if isinstance(pred, (int, np.integer)):
                    idx = int(pred)
                    prob = 1.0
                    return [{"intent": classes[idx], "probability": str(prob)}]
                else:
                    # assume label string
                    # find class index
                    if pred in classes:
                        return [{"intent": pred, "probability": "1.0"}]
                    else:
                        return [{"intent": "fallback", "probability": "1.0"}]
        except Exception:
            logging.error("Pickle model prediction failed:\n" + traceback.format_exc())
            raise
    else:
        raise RuntimeError("Unknown model type or no model loaded")

def get_response(intents_list):
    if not intents_list:
        return "I did not understand that."
    tag = intents_list[0]['intent']
    # check job_intents.json structure
    try:
        for intent in intents.get('intents', []):
            if intent.get('tag') == tag:
                return random.choice(intent.get('responses', ["I don't have an answer for that."]))
    except Exception:
        logging.error("Error while fetching response:\n" + traceback.format_exc())
    return "I'm not sure I understand you."

def chatbot_response(user_input: str) -> str:
    """
    Main entry. Returns a string response. Logs errors and returns an apologetic
    message on unexpected exceptions.
    """
    try:
        logging.info(f"chatbot_response called with: {user_input!r}")
        intents_pred = predict_intents(user_input)
        logging.info(f"Predicted intents: {intents_pred}")
        resp = get_response(intents_pred)
        logging.info(f"Returning response: {resp!r}")
        return resp
    except Exception:
        logging.error("Exception in chatbot_response:\n" + traceback.format_exc())
        return "Sorry, something went wrong!"
