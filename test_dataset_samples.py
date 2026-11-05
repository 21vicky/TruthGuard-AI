import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from text_cleaner import clean_text

# Load dataset and trained model
data = pd.read_csv("dataset/processed_news.csv")

model = joblib.load("models/fake_news_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

X = data["content"].fillna("")
y = data["label"]

# Recreate the same test split used during training
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Select 3 real and 3 fake articles from the test set
test_data = pd.DataFrame({
    "content": X_test,
    "actual": y_test
})

samples = pd.concat([
    test_data[test_data["actual"] == 0].head(3),
    test_data[test_data["actual"] == 1].head(3)
])

# Make predictions
cleaned_text = samples["content"].apply(clean_text)
vectors = tfidf.transform(cleaned_text)
predictions = model.predict(vectors)
probabilities = model.predict_proba(vectors)

# Display results
for i, (_, article) in enumerate(samples.iterrows()):
    actual = "REAL" if article["actual"] == 0 else "FAKE"
    predicted = "REAL" if predictions[i] == 0 else "FAKE"

    print("\n" + "=" * 60)
    print("ACTUAL:", actual)
    print("PREDICTED:", predicted)
    print("ARTICLE:", article["content"][:250], "...")

    print(
        "MODEL SCORES:",
        {
            "Real": round(probabilities[i][list(model.classes_).index(0)] * 100, 2),
            "Fake": round(probabilities[i][list(model.classes_).index(1)] * 100, 2)
        }
    )