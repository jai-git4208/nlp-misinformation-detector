#main pipeline for runnning clasifier model
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    print("loading data")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    try:

        true_df = pd.read_csv(os.path.join(DATASET_DIR, "True.csv"))
        fake_df = pd.read_csv(os.path.join(DATASET_DIR, "Fake.csv"))
        true_df['label'] = 1
        fake_df['label'] = 0

        df = pd.concat([true_df, fake_df])

        #shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None
    
import joblib

def save_model(model, vectorizer, model_path="model/model.pkl", vectorizer_path="model/vectorizer.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving model to {model_path} and vectorizer to {vectorizer_path}")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load_saved_model(model_path="model/model.pkl", vectorizer_path="model/vectorizer.pkl"): #windows users fuckin change the paths to \
    print("loading pre-trained model")
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    else:
        print("model not found xd")
        return None, None





def train_model(df, save=False):
    print('Splitting data')
    # we will use the 'text' column for classification
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Vectorizing text')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
    tfidf_test = tfidf_vectorizer.transform(X_test)

    print("Training PassiveAggressiveClassifier")
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    print('Evaluating model')

    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {score*100:.2f}%')

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if save:
        save_model(pac, tfidf_vectorizer)
    
    return pac, tfidf_vectorizer

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        model, vectorizer = train_model(df, save=True)
        print("Pipeline execution completed successfully. Model saved as well.")

