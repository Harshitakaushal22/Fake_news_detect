import zipfile
import pandas as pd

# Paths to the ZIP files
zip_fake_path = r"C:\Users\workk\Downloads\Fake.csv.zip"
zip_real_path = r"C:\Users\workk\Downloads\True.csv.zip"

# Directory where the files will be extracted
extract_dir = r"C:\Users\workk\Downloads"

# Function to extract a zip file
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# Extract the files
unzip_file(zip_fake_path, extract_dir)
unzip_file(zip_real_path, extract_dir)

# Now read the extracted CSV files
fake = pd.read_csv(r"C:\Users\workk\Downloads\Fake.csv")
real = pd.read_csv(r"C:\Users\workk\Downloads\True.csv")

# Check first few rows to confirm loading
print("Fake news data sample:")
print(fake.head())
print("\nReal news data sample:")
print(real.head())
import os
import zipfile
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Download nltk resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Paths
download_dir = r"C:\Users\workk\Downloads"
fake_csv_path = os.path.join(download_dir, "Fake.csv")
real_csv_path = os.path.join(download_dir, "True.csv")
fake_zip_path = os.path.join(download_dir, "Fake.csv.zip")
real_zip_path = os.path.join(download_dir, "True.csv.zip")

# Function to extract ZIP if CSV not present
def extract_if_needed(zip_path, csv_path):
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Extracting from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        print("Extraction done.")
    else:
        print(f"{csv_path} already exists. Skipping extraction.")

# Extract files if needed
extract_if_needed(fake_zip_path, fake_csv_path)
extract_if_needed(real_zip_path, real_csv_path)

# Load CSV files
fake = pd.read_csv(fake_csv_path)
real = pd.read_csv(real_csv_path)

# Label datasets and combine
fake['label'] = 0
real['label'] = 1
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
df = df[['text', 'label']]

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved!")

# Predict function for new text
def predict_news(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean]).toarray()
    pred = model.predict(vec)
    return "Real" if pred[0] == 1 else "Fake"

# Example usage
print(predict_news("The president announced a new economic plan today."))
print(predict_news("Breaking news! Celebrity caught in scandal."))
