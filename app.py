import os
import logging
import traceback
from flask import Flask, render_template, jsonify, request
import processor

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = (
            request.form.get('question')
            or (request.json and request.json.get('question'))
            or ''
        ).strip()

        if not user_input:
            return jsonify({"response": "Please enter a message."}), 400

        bot_response = processor.chatbot_response(user_input)
        logging.info(f"User: {user_input} | Bot: {bot_response}")
        return jsonify({"response": bot_response})

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error in chatbot route:\n{tb}")
        return jsonify({"response": "Sorry, something went wrong!"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
