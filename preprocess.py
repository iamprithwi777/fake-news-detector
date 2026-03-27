import pandas as pd
import string
import os


def load_and_preprocess(path):
    """
    Load + preprocess dataset
    """

    # Load CSV
    data = pd.read_csv(path, encoding='utf-8')

    print("✅ Data loaded successfully!")
    print("Columns:", data.columns.tolist())

    # Remove missing values
    data = data.dropna()

    # Combine title + text
    data['content'] = data['title'] + " " + data['text']

    # Clean text
    data['content'] = data['content'].apply(clean_text)

    # Keep required columns
    data = data[['content', 'label']]

    return data


def clean_text(text):
    """
    Lowercase + remove punctuation
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


if __name__ == "__main__":
    # Get correct path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'data', 'news.csv')

    print(f"📂 Loading file from: {file_path}")

    # Run preprocessing
    processed_data = load_and_preprocess(file_path)

    print("\n✅ Processed Data Preview:")
    print(processed_data.head())

    print("\n📊 Dataset shape:", processed_data.shape)