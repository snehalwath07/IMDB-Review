
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- NLTK Downloads (will only download if not present) ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

# --- Preprocessing Function ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove HTML tags (specific to IMDB dataset)
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    processed_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(processed_words)

# --- Main Model Building Steps ---
if __name__ == '__main__':
    print("Starting model building process...")

    # 1. Load Data
    # Make sure 'IMDB_Dataset.csv.zip' is in the same directory as this script
    try:
        df = pd.read_csv("IMDB_Dataset.csv.zip")
        print(f"Original DataFrame shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'IMDB_Dataset.csv.zip' not found. Please ensure it's in the same directory as the script.")
        exit()

    # 2. Data Cleaning: Remove Duplicate Rows
    df.drop_duplicates(inplace=True)
    print(f"DataFrame shape after removing duplicates: {df.shape}")

    # 3. Apply Preprocessing to Reviews
    print("Preprocessing text data...")
    df['processed_review'] = df['review'].apply(preprocess_text)

    # 4. Feature Extraction: Count Vectorization
    print("Performing Count Vectorization...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['processed_review'])
    print(f"Shape of feature matrix X: {X.shape}")
    print(f"Number of unique words (features): {len(vectorizer.get_feature_names_out())}")

    # 5. Preparing Target Variable and Splitting Data
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # 6. Hyperparameter Tuning for Logistic Regression
    print("Starting GridSearchCV for Logistic Regression... (This may take some time)")
    param_grid_lr = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga']
    }
    lr_grid_model = LogisticRegression(max_iter=1000, random_state=42)
    grid_search_lr = GridSearchCV(estimator=lr_grid_model, param_grid=param_grid_lr,
                                  cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search_lr.fit(X_train, y_train)

    best_lr_model = grid_search_lr.best_estimator_
    print(f"Best parameters found for Logistic Regression: {grid_search_lr.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search_lr.best_score_:.4f}")

    # 7. Model Evaluation
    best_lr_y_pred = best_lr_model.predict(X_test)
    best_lr_accuracy = accuracy_score(y_test, best_lr_y_pred)
    print(f"\nTest set accuracy with best Logistic Regression model: {best_lr_accuracy:.4f}")
    print("Classification Report for Best Logistic Regression Model:")
    print(classification_report(y_test, best_lr_y_pred))

    # 8. Save Model and Vectorizer
    joblib.dump(vectorizer, 'count_vectorizer.pkl')
    joblib.dump(best_lr_model, 'logistic_regression_model.pkl')
    print("Vectorizer and Logistic Regression model saved successfully to 'count_vectorizer.pkl' and 'logistic_regression_model.pkl'.")

    print("Model building process complete.")
