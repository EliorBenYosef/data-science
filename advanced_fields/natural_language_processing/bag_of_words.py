"""
Sentiment Analysis Task
Classifying reviews: Negative (0) / Positive (1)
"""

import pandas as pd
from csv import QUOTE_NONE
from utils import get_clean_corpus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from machine_learning.supervised_learning.classification.models_classification import ClassificationModels

# csv.QUOTE_NONE - ignore (remove) the quotes in the text
df = pd.read_csv('../../datasets/per_field/nlp/Restaurant_Reviews.tsv', delimiter='\t', quoting=QUOTE_NONE)
X = df.iloc[:, 0].values  # df['Review'].values
y = df.iloc[:, -1].values  # df['Liked'].values

##############################

# 1. Data Pre-Processing

# Cleaning the texts:
X = get_clean_corpus(X)

# Creating the Bag of Words model (data vectorization):
#   creating a sparse feature matrix containing the clean texts’ entire words count.

# # Option 1 - taking the 95% most frequent words:
# cv = CountVectorizer()
# n_words = cv.fit_transform(X).toarray().shape[1]
# print('number of unique words:', n_words)
# cv = CountVectorizer(max_features=round(n_words * 0.95))

# Option 2 - taking the words with a minimum count of 2:
cv = CountVectorizer(min_df=2)

X = cv.fit_transform(X).toarray()

##############################

# 2. Dataset Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

##############################

# 3. Classification (Response: No (0) / Yes (1)):
#   we can use a classical ML model or a deep learning model (NN)

classification_models = ClassificationModels(X_train, y_train, X_test, y_test)
classification_models.all()
classification_models.print_models_accuracy()

# Remember: Accuracy is not enough, so you should also look at other performance metrics like:
#   - Precision (measuring exactness)
#   - Recall (measuring completeness)
#   - F1 Score (compromise between Precision and Recall).

###############

new_reviews = ['I love this restaurant so much',
               'I hate this restaurant so much']
X_test = get_clean_corpus(new_reviews)
X_test = cv.transform(X_test).toarray()
for model_name, classifier in classification_models.classifiers.items():
    y_pred = classifier.predict(X_test)
    print(f'{model_name} predictions: {y_pred}')
