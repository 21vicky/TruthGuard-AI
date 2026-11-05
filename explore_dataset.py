import pandas as pd

# Load real news
true_news = pd.read_csv("dataset/True.csv")

# Load fake news
fake_news = pd.read_csv("dataset/Fake.csv")

# Display dataset information
print("REAL NEWS DATA")
print(true_news.head())

print("\nFAKE NEWS DATA")
print(fake_news.head())

# Count records
print("\nReal News Count:", len(true_news))
print("Fake News Count:", len(fake_news))

# Check columns
print("\nColumns:", true_news.columns.tolist())