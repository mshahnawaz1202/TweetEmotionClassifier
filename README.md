


# TweetEmotionClassifier

**TweetEmotionClassifier** is a Python NLP project for classifying emotions in tweets. It provides a complete pipeline from preprocessing text to feature extraction, class balancing, and train-test splitting for machine learning models.



## Features

- Upload Excel datasets directly in Google Colab.
- Preprocessing includes:
  - Tokenization
  - Stopword removal
  - Stemming
  - Lemmatization
- Supports multiple feature extraction techniques:
  - CountVectorizer
  - TF-IDF
  - Word2Vec embeddings
- Handles class imbalance using SMOTE.
- Generates train and test datasets ready for model training.



## Usage

1. Upload your Excel dataset containing tweets and emotion labels in the Colab notebook.
2. Run the preprocessing and feature extraction steps.
3. Use the generated train-test splits to train your classifier:

```python
# Example: Training with CountVectorizer features
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_count, y_balanced, test_size=0.2, random_state=42)
```

4. Train any ML classifier (Logistic Regression, Random Forest, etc.) and evaluate performance.

---

## Example Workflow

```python
# Preprocess tweets
df['Clean_Tweets'] = df['Tweets'].apply(preprocess_text)

# CountVectorizer features
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(df['Clean_Tweets'])

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_count, df['Level 2'], test_size=0.2, random_state=42)
```



## Requirements

* Python 3.x
* pandas
* scikit-learn
* nltk
* gensim
* imbalanced-learn
* openpyxl




## Author

**Shahnawaz**

