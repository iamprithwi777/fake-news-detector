from flask import Flask, request, render_template_string
import pickle
import os
import webbrowser
from threading import Timer

# ------------------ APP INIT ------------------
app = Flask(__name__)

# ------------------ PATH SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')


# ------------------ LOAD MODEL ------------------
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)

        print("✅ Model and vectorizer loaded successfully!")
        return model, vectorizer

    except Exception as e:
        print("❌ Error loading model:", e)
        exit()


model, vectorizer = load_model()


# ------------------ HTML TEMPLATE ------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            margin-top: 50px;
            background-color: #f4f4f4;
        }
        textarea {
            width: 60%;
            height: 120px;
            padding: 10px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            margin-top: 10px;
            font-size: 16px;
            cursor: pointer;
        }
        .result {
            font-size: 22px;
            margin-top: 20px;
            font-weight: bold;
        }
        .real { color: green; }
        .fake { color: red; }
    </style>
</head>
<body>

    <h1>📰 Fake News Detector</h1>

    <form method="POST">
        <textarea name="news" placeholder="Enter news text here..." required></textarea><br>
        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
        <div class="result {{ prediction_class }}">
            Prediction: {{ prediction }}
        </div>
    {% endif %}

</body>
</html>
"""


# ------------------ ROUTES ------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prediction_class = ""

    if request.method == "POST":
        text = request.form.get("news")

        if text:
            text_vec = vectorizer.transform([text])
            result = model.predict(text_vec)[0]

            prediction = result
            prediction_class = "real" if result == "REAL" else "fake"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        prediction_class=prediction_class
    )


# ------------------ AUTO OPEN BROWSER ------------------
def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


# ------------------ RUN APP ------------------
if __name__ == "__main__":
    Timer(1, open_browser).start()  # opens browser after 1 sec
    app.run(debug=True)