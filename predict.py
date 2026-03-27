import pickle
import os


def load_model():
    """
    Load trained model and vectorizer
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, 'model.pkl')
    vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_news(text, model, vectorizer):
    """
    Predict REAL or FAKE
    """
    # Convert to vector
    text_vec = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vec)[0]

    return prediction


if __name__ == "__main__":
    print("📰 Fake News Detector")
    print("Type 'exit' to quit\n")

    # Load model
    model, vectorizer = load_model()

    while True:
        user_input = input("Enter news text: ")

        if user_input.lower() == "exit":
            break

        result = predict_news(user_input, model, vectorizer)

        print(f"🔍 Prediction: {result}\n")