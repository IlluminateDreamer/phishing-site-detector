# -*- coding: utf-8 -*-
"""
Phishing Site Prediction
A machine learning model to detect phishing websites based on their URLs.
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

# Loading the dataset
df = pd.read_csv("phishing_site_urls.csv")

# Initial exploration of the dataset
print("First few rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nShape of the dataset:", df.shape)
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualizing the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x="Label", data=df)
plt.title("Distribution of Good and Bad Sites")
plt.show()

# Preprocessing the data
print("\nPreprocessing the data...")

# Tokenization using RegexpTokenizer
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
df['text_tokenized'] = df['URL'].map(lambda url: tokenizer.tokenize(url))

# Stemming using SnowballStemmer
stemmer = SnowballStemmer("english")
df['text_stemmed'] = df['text_tokenized'].map(lambda words: [stemmer.stem(word) for word in words])

# Joining the stemmed words into a single string
df['text_sent'] = df['text_stemmed'].map(lambda words: ' '.join(words))

# Feature extraction using TfidfVectorizer for better performance
vectorizer = TfidfVectorizer(tokenizer=RegexpTokenizer(r'[A-Za-z]+').tokenize, stop_words='english')
X = vectorizer.fit_transform(df['text_sent'])
y = df['Label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training and evaluation
print("\nTraining Logistic Regression model...")
lr = LogisticRegression(max_iter=1000, C=1.0, penalty='l2')
lr.fit(X_train, y_train)

# Evaluating the Logistic Regression model
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Model Evaluation:")
print("Training Accuracy:", lr.score(X_train, y_train))
print("Testing Accuracy:", lr.score(X_test, y_test))
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=['Good', 'Bad']))

# Confusion matrix visualization for Logistic Regression
conf_matrix_lr = pd.DataFrame(confusion_matrix(y_test, y_pred_lr),
                              columns=['Predicted:Good', 'Predicted:Bad'],
                              index=['Actual:Good', 'Actual:Bad'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

# Training and evaluating Multinomial Naive Bayes model
print("\nTraining Multinomial Naive Bayes model...")
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Evaluating the Multinomial Naive Bayes model
y_pred_mnb = mnb.predict(X_test)
print("\nMultinomial Naive Bayes Model Evaluation:")
print("Training Accuracy:", mnb.score(X_train, y_train))
print("Testing Accuracy:", mnb.score(X_test, y_test))
print("\nClassification Report for Multinomial Naive Bayes:")
print(classification_report(y_test, y_pred_mnb, target_names=['Good', 'Bad']))

# Confusion matrix visualization for Multinomial Naive Bayes
conf_matrix_mnb = pd.DataFrame(confusion_matrix(y_test, y_pred_mnb),
                               columns=['Predicted:Good', 'Predicted:Bad'],
                               index=['Actual:Good', 'Actual:Bad'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_mnb, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix for Multinomial Naive Bayes")
plt.show()

# Comparing the two models' accuracies
scores_ml = {
    'Logistic Regression': np.round(lr.score(X_test, y_test), 2),
    'Multinomial Naive Bayes': np.round(mnb.score(X_test, y_test), 2)
}
accuracy_df = pd.DataFrame.from_dict(scores_ml, orient='index', columns=['Accuracy'])
sns.barplot(x=accuracy_df.index, y=accuracy_df['Accuracy'])
plt.title("Model Comparison: Accuracy")
plt.show()

# Saving the best-performing model (Logistic Regression) with pickle
print("\nSaving the best model...")
pipeline_lr = make_pipeline(vectorizer, lr)
pickle.dump(pipeline_lr, open('phishing_model.pkl', 'wb'))

# Function for making predictions
def predict_phishing(url):
    """
    Predict whether a given URL is phishing or safe.
    Parameters:
        url (str): The URL to be classified.
    Returns:
        str: "Phishing" if the URL is vulnerable, "Safe" otherwise.
    """
    prediction = pipeline_lr.predict([url])
    return "Phishing" if prediction[0] == 'bad' else "Safe"

# Example usage
url_to_check = "http://google.com"
result = predict_phishing(url_to_check)
print(f"\nThe URL '{url_to_check}' is classified as: {result}")
