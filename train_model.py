import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(path):
    """
    Load dataset
    """
    data = pd.read_csv(path, encoding='utf-8')
    return data


def preprocess(data):
    """
    Same preprocessing (important)
    """
    data = data.dropna()
    data['content'] = data['title'] + " " + data['text']
    return data[['content', 'label']]


if __name__ == "__main__":
    # 📂 Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'news.csv')

    print("📂 Loading data...")
    data = load_data(data_path)

    print("⚙️ Preprocessing...")
    data = preprocess(data)

    # Split data
    X = data['content']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF
    print("🔢 Converting text to vectors...")
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    print("🤖 Training model...")
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Prediction
    y_pred = model.predict(X_test_vec)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc * 100:.2f}%")

    # 💾 Save model
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print("💾 Model and vectorizer saved!")