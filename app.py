import os
import logging
import traceback
from flask import Flask, render_template, jsonify, request
import processor

# Initialize Flask app
app = Flask(__name__)

# Set up logging
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
        # Accept both JSON and form data
        user_input = request.form.get('question') or (request.json and request.json.get('question')) or ''
        if not user_input.strip():
            return jsonify({"response": "Please type something."}), 400

        logging.info(f"User input: {user_input}")

        bot_response = processor.chatbot_response(user_input)
        logging.info(f"Bot response: {bot_response}")

        return jsonify({"response": bot_response})

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error in chatbot route:\n{tb}")
        return jsonify({"response": f"Sorry, something went wrong on server: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
