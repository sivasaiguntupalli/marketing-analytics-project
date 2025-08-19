import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text: str) -> str:
    '''
    Basic text cleaning: lowercasing, removing non-alphabetic characters.
    '''
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def train_sentiment_model(df: pd.DataFrame, text_col: str = "ReviewText", label_col: str = "Rating", positive_threshold: int = 4) -> tuple:
    '''
    Train a simple sentiment classifier using TF-IDF features and logistic regression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing review texts and ratings.
    text_col : str
        Name of the text column.
    label_col : str
        Name of the rating column.
    positive_threshold : int
        Ratings equal to or above this threshold are considered positive.

    Returns
    -------
    tuple
        The trained vectorizer and model.
    '''
    # Define binary labels: positive if rating >= threshold
    df = df.copy()
    df["label"] = (df[label_col] >= positive_threshold).astype(int)
    # Clean text
    df["cleaned_text"] = df[text_col].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print("Sentiment Model Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return vectorizer, model
