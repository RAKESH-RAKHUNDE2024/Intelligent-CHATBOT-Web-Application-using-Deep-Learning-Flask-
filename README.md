# 🤖 Intelligent CHATBOT Web Application Using Deep Learning & Flask

### Project Summary
This Project Is An Interactive **AI Powered CHATBOT** Built With **Flask** And **TensorFlow**.  
It Uses A Trained Deep Learning Model To Understand User Inputs And Respond Conversationally Through A Simple And Responsive Web Interface.  
The Model Is Loaded Using Keras, While Flask Serves As The Backend Framework To Handle API Requests And Render The Frontend Chat UI.

---

### 🎯 Objective
- Develop An End To End CHATBOT Capable Of Responding Intelligently To User Queries.  
- Deploy The CHATBOT As A **Flask Web Application** That Can Run Both Locally And On The Cloud (Render).  
- Demonstrate Model Integration, Data Reprocessing, And Real-Time Response Handling.  
- Build A Simple, Clean UI Using **HTML (Jinja2 Templates)** For User Interaction.

---

### 🧠 Key Insights
- The CHATBOT Leverages A Trained **Tensorflow/Keras Model (`chatbot_model.h5`)** For Text Classification And Intent Recognition.  
- **NLTK** Is Used For Natural Language Reprocessing Such As Tokenization And Stemming.  
- The App Exposes An Endpoint (`/CHATBOT`) To Receive User Queries And Return Responses As JSON.  
- The Project Demonstrates A Full ML Lifecycle: Model Loading → Inference → Flask Integration → Web Deployment.

---

### 🧰 Technologies & Libraries Used
| Category | Libraries/Tools |
|-----------|----------------|
| **Backend Framework** | Flask |
| **Machine Learning / DL** | Tensorflow / Keras |
| **NLP Processing** | NLTK |
| **Data Handling** | NumPy |
| **Deployment** | Render / GitHub |
| **Web UI** | HTML, CSS (Flask Templates) |

---

### 📂 Project Structure
CHATBOT_Project/

│
├── app.py # Main Flask Application File

├── processor.Py # CHATBOT Logic & Model Loading

├── CHATBOT_Model.h5 # Pretrained Tensorflow Model

├── Requirements.txt # List Of Required Python Packages

├── Procfile # Render Deployment Instruction

├── .Gitignore # Files And Folders To Ignore In Git

│
├── Templates/

│ └── Index.Html # Frontend HTML Interface For CHATBOT

│
└── Pycache/ # Auto Generated Cache Files (Ignored)



---

### 📊 Example Interaction

User: “Hello!”

CHATBOT: “Hi There! How Can I Help You Today?”

User: “Tell Me About Your Features.”

CHATBOT: “I’m Designed To Assist You With Basic Queries And Automate Responses Using Deep Learning Models.”

### 🏁 Conclusion

This Project Demonstrates How Deep Learning Models Can Be Integrated With Flask To Create Scalable, Production Ready Chat Applications.
It Also Highlights The Process Of Deploying AI Models On The Web, Making Machine Learning Solutions More Accessible To End Users.

### 👨💻 Author

Developed By: Rakesh N. Rakhunde

Role: Data Engineer / Data Science Enthusiast

Tech Stack: Python | SQL | Power BI | Flask | Machine Learning | Deep Learning
