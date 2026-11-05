
import joblib

# Load trained model
model = joblib.load("models/fake_news_model.pkl")

# Load TF-IDF vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def predict_news(news):

    # Convert input text into numerical features
    news_vector = tfidf.transform([news])

    # Predict news category
    prediction = model.predict(news_vector)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(news_vector)[0]

    # Get confidence of predicted class
    confidence = max(probabilities) * 100

    if prediction == 0:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return result, confidence


# Test prediction
if __name__ == "__main__":

    news = input("Enter a news headline or article: ")

    result, confidence = predict_news(news)

    print("\nPrediction:", result)
    print("Model confidence:", round(confidence, 2), "%")
    print("\nNote: This is a model prediction, not verified fact-checking.")
