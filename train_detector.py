import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1) Load
df = pd.read_csv("AI_Human.csv", low_memory=False)

# 2) Clean columns
df.columns = [c.strip() for c in df.columns]
print("Columns found:", df.columns.tolist())

# 3) Remove junk columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 4) Keep only required columns + drop empty rows
df = df[["text", "generated"]].dropna()
df = df[df["generated"].isin([0, 1])]

print(df["generated"].value_counts())
print(df.head(2))

# 5) (Recommended) sample for faster training while developing
df = df.sample(100000, random_state=42)

X = df["text"]
y = df["generated"]

# 6) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8) Train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# 9) Evaluate
y_pred = model.predict(X_test_vec)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 10) Save model + vectorizer
joblib.dump(model, "ai_detector_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nSaved: ai_detector_model.pkl and tfidf_vectorizer.pkl")
