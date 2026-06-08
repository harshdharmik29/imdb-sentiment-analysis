# 🎬 IMDB Sentiment Analysis

A machine learning project that analyzes movie reviews and predicts whether the sentiment of a review is positive or negative. The project uses Natural Language Processing (NLP) techniques for text preprocessing and classification.

## Overview

The main goal of this project is to understand how text data can be processed and used for sentiment classification. The IMDB movie review dataset was used to train and evaluate multiple machine learning models.

The application is deployed using Streamlit, allowing users to enter a movie review and instantly get a sentiment prediction.

## Features

* Text preprocessing and cleaning
* Tokenization
* Stopword removal
* Lemmatization
* TF-IDF vectorization
* Sentiment prediction
* Interactive Streamlit web application
* Model comparison and evaluation

## Dataset

The project uses the IMDB Movie Reviews Dataset containing positive and negative movie reviews.

## Workflow

1. Load and explore the dataset
2. Handle missing values and duplicates
3. Clean review text
4. Perform NLP preprocessing
5. Convert text into numerical features using TF-IDF
6. Split data into training and testing sets
7. Train multiple machine learning models
8. Evaluate model performance
9. Select the best-performing model
10. Deploy using Streamlit

## Models Used

* Logistic Regression
* Naive Bayes
* Random Forest

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* Streamlit
* GitHub

## Project Structure

```text
IMDB-Sentiment-Analysis
│
├── app.py
├── model.pkl
├── tfidf.pkl
├── requirements.txt
└── README.md
```

## Running the Project Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## Sample Reviews

Positive Review:

```text
This movie was amazing. The acting and storyline were excellent.
```

Negative Review:

```text
The movie was boring and a complete waste of time.
```

## Learning Outcomes

Through this project, I gained practical experience in:

* Text preprocessing techniques
* Feature extraction using TF-IDF
* Machine learning model training
* Model evaluation and comparison
* Deploying ML applications using Streamlit

## Author

Harsh Dharmik

This project was created as part of my learning journey in Machine Learning and Natural Language Processing.
