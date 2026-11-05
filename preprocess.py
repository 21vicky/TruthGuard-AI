from text_cleaner import clean_text
import pandas as pd
import re

# Load datasets
true_news = pd.read_csv("dataset/True.csv")
fake_news = pd.read_csv("dataset/Fake.csv")

# Assign labels
true_news["label"] = 0  # Real news
fake_news["label"] = 1  # Fake news

# Combine both datasets
data = pd.concat([true_news, fake_news], ignore_index=True)

# Handle missing values
data = data.dropna(subset=["title", "text"])

# Combine headline and article
data["content"] = data["title"] + " " + data["text"]


# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply text cleaning
data["content"] = data["content"].apply(clean_text)

# Remove empty content
data = data[data["content"] != ""]

# Remove duplicate articles
data = data.drop_duplicates(subset=["content"])

# Keep only required columns
data = data[["content", "label"]]

# Save processed dataset
data.to_csv("dataset/processed_news.csv", index=False)

# Display results
print("PREPROCESSING COMPLETED!")
print("--------------------------------")

print("Total Articles:", len(data))
print("Real News:", len(data[data["label"] == 0]))
print("Fake News:", len(data[data["label"] == 1]))

print("\nSample Cleaned News:")
print(data["content"].iloc[0][:500])

print("\nProcessed dataset saved successfully!")
