"""
Sentiment Analysis Task
Classifying reviews: Negative (0) / Positive (1)
"""

import pandas as pd
from csv import QUOTE_NONE
from advanced_fields.natural_language_processing.utils import TextCleaner
from advanced_fields.natural_language_processing.models.text_vectorization import bag_of_words
from sklearn.model_selection import train_test_split
from machine_learning.supervised_learning.classification.models_classification import ClassificationModels

# csv.QUOTE_NONE - ignore (remove) the quotes in the text
df = pd.read_csv('../../../datasets/per_field/nlp/labeled_reviews.tsv', delimiter='\t', quoting=QUOTE_NONE)
cat_var = 'Liked'  # No (0) / Yes (1)
text_var = 'Review'

##############################

# 1. Data Pre-Processing

# Text Cleaning:
tc = TextCleaner()
tc.stop_words.remove('not')  # crucial for sentiment analysis
df[text_var + '_clean'] = df[text_var].apply(lambda x: tc.clean_review(x))

# Text Vectorization:
X = df[text_var + '_clean'].values
X, word_vectorizer = bag_of_words(X)

y = df[cat_var].values

##############################

# 2. Dataset Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##############################

# 3. Binary Classification:
#   we can use a classical ML model or a deep learning model (NN)

classification_models = ClassificationModels(X_train, y_train, X_test, y_test)
classification_models.all()
print()
classification_models.print_models_accuracy()

##############################

print()
new_reviews = ['I love this restaurant so much',
               'I hate this restaurant so much']
X_test = [tc.clean_review(x) for x in new_reviews]
X_test = word_vectorizer.transform(X_test).toarray()
for model_name, classifier in classification_models.classifiers.items():
    y_pred = classifier.predict(X_test)
    print(f'{model_name} predictions: {y_pred}')
