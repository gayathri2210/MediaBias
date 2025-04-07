# Import libraries first
# Add this at the very top of your imports
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('wordnet')

# Then keep your existing code
from textblob import TextBlob
import matplotlib.pyplot as plt

import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# Define the function BEFORE calling it
def initialize_keybert():
    try:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        return KeyBERT(model=sbert_model)
    except Exception as e:
        print(f"Model load failed: {e}")
        return KeyBERT()  # Fallback to default

# Now call the function
kw_model = initialize_keybert()

# Rest of your code (e.g., loading CSV, processing data)
df = pd.read_csv('scraped_articles.csv')

# Clean data: Remove rows with missing or short phrases
df = df.dropna(subset=['meaningful_phrase'])
df = df[df['meaningful_phrase'].str.len() > 3]

# 1. Sentiment Analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns sentiment polarity (-1 to 1)

df['sentiment'] = df['meaningful_phrase'].apply(get_sentiment)

# 2. Bias Scoring using KeyBERT confidence values
def bias_score(text):
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            top_n=3,
            stop_words='english'
        )
        return sum([kw[1] for kw in keywords]) / len(keywords)  # Average confidence score
    except:
        return 0

df['bias_score'] = df['meaningful_phrase'].apply(bias_score)

# 3. Political Lean Normalization (left as negative, right as positive)
def analyze_leaning(row):
    if row['bias'] == 'left':
        return row['bias_score'] * -1  # Left bias as negative scores
    else:
        return row['bias_score']  # Right bias as positive scores

df['lean_score'] = df.apply(analyze_leaning, axis=1)

# Save the updated DataFrame to a new CSV file for reference
df.to_csv('analyzed_articles.csv', index=False)

# 4. Visualization: Sentiment and Bias Analysis

# Sentiment distribution by bias group (left vs right)
plt.figure(figsize=(12, 6))
for bias_group in ['left', 'right']:
    subset = df[df['bias'] == bias_group]
    subset['sentiment'].plot(kind='kde', label=bias_group)
plt.title('Sentiment Distribution by Bias Group')
plt.xlabel('Sentiment Polarity')
plt.legend()
plt.savefig('sentiment_distribution.png')
plt.show()

# Average bias score comparison by outlet group (bar chart)
plt.figure(figsize=(8, 6))
df.groupby('bias')['bias_score'].mean().plot(kind='bar', color=['blue', 'red'])
plt.title('Average Bias Score by Outlet Group')
plt.ylabel('Average Bias Score')
plt.xticks(rotation=0)
plt.savefig('bias_score_comparison.png')
plt.show()

# 5. Top Biased Phrases: Extract most biased phrases per group
def top_phrases(group, n=5):
    return group.sort_values('bias_score', ascending=False).head(n)

left_phrases = top_phrases(df[df['bias'] == 'left'])
right_phrases = top_phrases(df[df['bias'] == 'right'])

print("\nTop Left-Leaning Phrases:")
print(left_phrases[['meaningful_phrase', 'bias_score']])
print("\nTop Right-Leaning Phrases:")
print(right_phrases[['meaningful_phrase', 'bias_score']])

print("\nAnalysis Complete! Results saved to 'analyzed_articles.csv'")