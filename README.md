# TweetEmotionClassifier

A comprehensive NLP pipeline for classifying emotions in tweets using multiple machine learning approaches and feature extraction techniques.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Example Workflow](#example-workflow)
- [Model Performance](#model-performance)
- [Author](#author)

---

## ✨ Features

- **Data Upload**: Upload Excel datasets directly in Google Colab
- **Text Preprocessing**: 
  - Tokenization
  - Stopword removal
  - Stemming & Lemmatization
- **Multiple Vectorization Techniques**:
  - CountVectorizer
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Word2Vec embeddings
- **Class Balancing**: RandomOverSampler for handling imbalanced datasets
- **Automated Model Selection**: Test 7+ classifiers to identify best performers
- **Comprehensive Evaluation**: 9 complete experiments (3 models × 3 vectorization techniques)
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrices
- **Visualization**: Class distribution plots and confusion matrix heatmaps

---

## 📊 Project Overview

This pipeline implements a complete machine learning workflow:

1. **Data Loading** - Import Excel files with tweets and emotion labels
2. **Preprocessing** - Clean and normalize text data
3. **Feature Engineering** - Extract features using three different methods
4. **Class Balancing** - Handle imbalanced emotion classes
5. **Model Training** - Train multiple classifiers
6. **Evaluation** - Comprehensive performance analysis
7. **Visualization** - Charts and confusion matrices

---

## 📦 Requirements

```
Python 3.x
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
imbalanced-learn
gensim
openpyxl
gspread
gspread_pandas
```

### Install via pip:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn imbalanced-learn gensim openpyxl
pip install --quiet gspread gspread_pandas --upgrade
```

---

## 🚀 Installation

### Setup in Google Colab:

```python
# Install required packages
!pip install pandas numpy matplotlib seaborn nltk scikit-learn imbalanced-learn gensim
!pip install --quiet gspread gspread_pandas --upgrade

# Download NLTK resources
!python -m nltk.downloader punkt punkt_tab stopwords wordnet
```

### Local Setup:

1. Clone or download the project
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook or Python script

---

## 🔄 Pipeline Architecture

```
Raw Data (Excel)
       ↓
Text Preprocessing
(Tokenization, Stopword Removal, Stemming, Lemmatization)
       ↓
Feature Extraction
(CountVectorizer | TF-IDF | Word2Vec)
       ↓
Class Balancing
(RandomOverSampler)
       ↓
Train-Test Split (80-20)
       ↓
Model Training & Evaluation
(7+ Classifiers × 3 Vectorization Techniques)
       ↓
Performance Analysis & Visualization
```

---

## 💻 Usage

### Step 1: Data Upload

```python
from google.colab import files
import pandas as pd

# Upload Excel file
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_excel(file_name)

# Load data
df = df.dropna(subset=['Tweets', 'Level 2'])
X_raw = df['Tweets'].astype(str)
y_raw = df['Level 2']
```

### Step 2: Text Preprocessing

```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenization and lowercasing
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    
    # Stopword removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming and Lemmatization
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

df['Clean_Tweets'] = X_raw.apply(preprocess_text)
```

### Step 3: Feature Vectorization

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# CountVectorizer
count_vec = CountVectorizer(max_features=5000)
X_count = count_vec.fit_transform(df['Clean_Tweets'])

# TF-IDF
tfidf_vec = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vec.fit_transform(df['Clean_Tweets'])

# Word2Vec
tokenized = [text.split() for text in df['Clean_Tweets']]
w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2)
```

### Step 4: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_count, 
    df['Level 2'], 
    test_size=0.2, 
    random_state=42
)
```

### Step 5: Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Train model
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
```

---

## 📈 Results

### Model Selection Results (Top Performers on TF-IDF)

| Model | Accuracy |
|-------|----------|
| Random Forest | 0.9463 |
| Decision Tree | 0.9142 |
| KNN | 0.8870 |
| SVC (Linear) | 0.8770 |
| Logistic Regression | 0.8645 |
| Multinomial NB | 0.7939 |
| AdaBoost | 0.3430 |

### Comprehensive Experiment Results

| Model | Vectorization | Accuracy | Precision | Recall | F1-Score |
|-------|---|----------|-----------|--------|----------|
| Random Forest | CountVectorization | 0.9503 | 0.9504 | 0.9503 | 0.9495 |
| Decision Tree | CountVectorization | 0.9158 | 0.9174 | 0.9158 | 0.9102 |
| KNN | CountVectorization | 0.8737 | 0.8707 | 0.8737 | 0.8684 |
| Random Forest | TF-IDF | 0.9463 | 0.9463 | 0.9463 | 0.9453 |
| Decision Tree | TF-IDF | 0.9142 | 0.9161 | 0.9142 | 0.9083 |
| KNN | TF-IDF | 0.8870 | 0.8856 | 0.8870 | 0.8851 |
| Random Forest | Word2Vec | 0.9434 | 0.9436 | 0.9434 | 0.9422 |
| Decision Tree | Word2Vec | 0.9103 | 0.9118 | 0.9103 | 0.9041 |
| KNN | Word2Vec | 0.8664 | 0.8727 | 0.8664 | 0.8536 |

**Best Overall**: Random Forest with CountVectorization (95.03% accuracy)

---

## 📁 Project Structure

```
CP_DS/
├── README.md
├── nlp_pipeline_ds.ipynb          # Main Jupyter notebook
├── nlp_pipeline.ipynb             # Alternative notebook
├── nlp_pipeline.py                # Python script version
├── experiment_results.csv         # Complete experiment results
├── automated_model_selection_results.csv
└── data/
    └── tweets_emotions.xlsx       # Sample dataset (user-provided)
```

---

## 🎯 Example Workflow

### Complete End-to-End Example:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
df = pd.read_excel('tweets_emotions.xlsx')
df = df.dropna(subset=['Tweets', 'Level 2'])

# 2. Preprocess
df['Clean_Tweets'] = df['Tweets'].apply(preprocess_text)

# 3. Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Clean_Tweets'])
y = df['Level 2']

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train & Evaluate
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

## 🏆 Model Performance

### Key Insights:

1. **Best Model**: Random Forest consistently outperforms other classifiers across all vectorization methods
2. **Best Vectorization**: CountVectorization achieves the highest accuracy (95.03%)
3. **Trade-offs**: 
   - CountVectorizer: High accuracy but sparse features
   - TF-IDF: Balanced performance with weighted term importance
   - Word2Vec: Lower performance but semantic context preservation

### Confusion Matrix Interpretation:

- **True Positives**: Correctly classified emotions
- **False Positives**: Incorrectly classified as a specific emotion
- **False Negatives**: Missed emotional classifications
- **True Negatives**: Correctly identified non-occurrences

---

## 🔍 Preprocessing Details

The text preprocessing pipeline includes:

1. **Tokenization**: Break text into individual words
2. **Lowercasing**: Convert all text to lowercase for consistency
3. **Stopword Removal**: Remove common English words (the, is, etc.)
4. **Alphabetic Filtering**: Keep only alphabetic characters
5. **Stemming**: Reduce words to root form (running → run)
6. **Lemmatization**: Convert words to dictionary form

Example:
```
Original:  "I'm really feeling happy and excited!!!"
Processed: "feel happi excit"
```

---

## 📊 Vectorization Techniques Explained

### CountVectorizer
- Counts word occurrences in each document
- Produces sparse matrices
- Best for: Document frequency-based analysis

### TF-IDF
- Weights terms by importance (Term Frequency × Inverse Document Frequency)
- Reduces impact of common terms
- Best for: Balanced feature importance

### Word2Vec
- Creates dense word embeddings
- Captures semantic relationships
- Best for: Understanding word context and meaning

---

## 🛠️ Troubleshooting

### Issue: NLTK data not found
**Solution**: Run `nltk.download('punkt')` and required packages individually

### Issue: Memory error with large datasets
**Solution**: Reduce `max_features` in vectorizers or use chunking

### Issue: Class imbalance warnings
**Solution**: Use `RandomOverSampler` or `random_state` in train_test_split

---

## 📝 Output Files

The pipeline generates:
- `experiment_results.csv` - All experiment metrics
- `automated_model_selection_results.csv` - Model selection results
- Confusion matrix visualizations (PNG)
- Class distribution charts (PNG)

---

## 🎓 Learning Outcomes

After working with this pipeline, you'll understand:
- Text preprocessing and NLP fundamentals
- Multiple feature extraction techniques
- Model selection and evaluation strategies
- How to handle imbalanced datasets
- Performance metrics interpretation
- End-to-end machine learning workflow

---

## 📚 References

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

---

## 👤 Author

**Shahnawaz**

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new classifiers or vectorization techniques
- Improve documentation

---

## 💡 Future Enhancements

- [ ] Add BERT/Transformer-based embeddings
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Deploy as web API
- [ ] Add real-time prediction functionality
- [ ] Support for multilingual datasets


