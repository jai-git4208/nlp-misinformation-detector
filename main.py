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
def train_model(df):
    print('splitij data1')
    #we will use the 'text' column for classsificatioon
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('vectorising text')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
    tfidf_test = tfidf_vectorizer.transform(X_test)

    print("training PassiveAgressiveClassifier")
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    print('evaluating model')

    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'accuracy: {score*100:.2f}%')

    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    
    print("\nclassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pac, tfidf_vectorizer

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        model, vectorizer = train_model(df)
        print("pipeline execution completed successfully.")

