import os
import logging
import traceback
import nltk
import numpy as np
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)

# Ensure NLTK data is available
for pkg in ['punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

# Load model safely
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'chatbot_model.h5')
model = None

try:
    if os.path.exists(MODEL_PATH):
        logging.info(f"Model found at: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        logging.info("✅ Chatbot model loaded successfully.")
    else:
        logging.error(f"❌ Model not found at {MODEL_PATH}")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    logging.error(traceback.format_exc())

# Dummy function (replace with your real preprocessing + model prediction)
def chatbot_response(user_input):
    try:
        if model is None:
            raise FileNotFoundError("Model not loaded or missing")

        # Replace below with your real model inference logic
        # Example (you can adapt as per your real code)
        # preprocessed_input = preprocess(user_input)
        # prediction = model.predict(preprocessed_input)
        # response = decode_prediction(prediction)

        # Temporary test response:
        return f"Bot received: {user_input}"

    except Exception as e:
        logging.error(f"Error in chatbot_response:\n{traceback.format_exc()}")
        return "Sorry, something went wrong!"
