# train_model.py

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

######################################
# PART ONE: Data Prep & Model Training

# Step One: Kaggle data downloaded
# Step Two: Create this training script 

# Step Three: Load & Preprocess Data 

# load data into pandas dataframe
df = pd.read_csv("IMDB Dataset.csv")

# Features (X = review) and labels (y = sentiment)
X = df["review"]
y = df["sentiment"]

# Step Four: Train the Model

# Create a pipeline that first transforms the text data using TfidfVectorizer 
# and then feeds it to the MultinomialNB classifier.

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
    ])

# train the model on our entire dataset 

model.fit(X, y)


# Step Five: Save the Model Pipeline

# Use the joblib library to dump your trained Pipeline object into a file named sentiment_model.pkl.

joblib.dump(model, "sentiment_model.pkl")

print("Part One Complete")