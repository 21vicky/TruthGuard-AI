
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# 1. Load processed dataset
data = pd.read_csv("dataset/processed_news.csv")

X = data["content"].fillna("")
y = data["label"]

print("Dataset loaded successfully!")
print("Total articles:", len(data))
print("Real articles:", (y == 0).sum())
print("Fake articles:", (y == 1).sum())

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Convert text into TF-IDF features
tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF features:", X_train_tfidf.shape[1])

# 4. Create Logistic Regression model
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

# 5. Three-fold cross-validation
print("\nRunning 3-fold cross-validation...")

cv_scores = cross_val_score(
    model,
    X_train_tfidf,
    y_train,
    cv=3,
    scoring="f1_macro"
)

print("Cross-validation F1 scores:", cv_scores)
print("Average CV F1:", round(cv_scores.mean() * 100, 2), "%")

# 6. Train final model
print("\nTraining final model...")

model.fit(X_train_tfidf, y_train)

# 7. Evaluate on unseen test split
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\nMODEL TRAINING COMPLETED!")
print("-------------------------")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("Macro F1:", round(macro_f1 * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["REAL NEWS", "FAKE NEWS"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 8. Save trained model and vectorizer
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
