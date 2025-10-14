# app.py (snippet)
import os
import logging
import traceback
from flask import Flask, render_template, jsonify, request

import processor

app = Flask(__name__)

# Logging to file and stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbotResponse():
    try:
        # Accept either form or JSON
        user_input = request.form.get('question') or (request.json and request.json.get('question')) or ''
        logging.info(f"Incoming question: {user_input!r}")

        if not user_input:
            return jsonify({"response": "Please enter a question"}), 400

        response = processor.chatbot_response(user_input)
        logging.info(f"Response: {response!r}")
        return jsonify({"response": response})

    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Error in /chatbot endpoint:\n" + tb)
        # Return a helpful but short error message to client; full trace is in logs.
        return jsonify({"response": "Sorry, something went wrong!", "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
