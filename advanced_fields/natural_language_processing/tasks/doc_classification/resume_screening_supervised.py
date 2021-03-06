"""
(Supervised) Resume Screening (Classification) Task (into job categories) via Text Vectorization.
1. Text Cleaning
1. Text Vectorization
2. Multiclass Classification (Supervised) - classifying labeled resumes into job categories

Dataset shape: (962, 2)
Dataset columns:
- Category: job type.
- Resume: resume.
"""

import pandas as pd
from data_tools.data_analyzing_tools import analyze_df, analyze_cat_var
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from advanced_fields.natural_language_processing.utils import TextCleaner
from advanced_fields.natural_language_processing.models.text_vectorization import Vectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_csv('../../../../datasets/per_field/nlp/labeled_resumes.csv', encoding='utf-8')
cat_var = 'Category'
text_var = 'Resume'

###################################

# # Exploratory Data Analysis:
# analyze_df(resumeDataSet)
# analyze_cat_var(resumeDataSet, cat_var, horizontal=False, show=False)

###################################

# Data Pre-Processing

# Encoding categorical data
#   Encoding the cat_var column using LabelEncoding
#       (since it's our target label column, even though it contains ‘Nominal’ data)
#   each category will become a class and we will be building a multiclass classification model.
y = df[cat_var].values
le = LabelEncoder()
y = le.fit_transform(y)

remove_stopwords_manually = True

# Text Cleaning:
tc = TextCleaner(remove_stopwords_manually)
df[text_var + '_clean'] = df[text_var].apply(lambda x: tc.clean_resume(x))

X = df[text_var + '_clean'].values

##############################

# Dataset Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##############################

# Data Pre-Processing

# Text Vectorization:
vactorizer = Vectorizer(X_train, max_features=1500, stopwords_removed_manually=remove_stopwords_manually)
X_train = vactorizer.tf_idf(sublinear_tf=True)
X_test = vactorizer.word_vectorizer.transform(X_test).toarray()

##############################

# Multiclass Classification:
#   building the model using: ‘One vs Rest’ method using a ‘KNeighborsClassifier’ model as an estimator.

classifier = OneVsRestClassifier(estimator=KNeighborsClassifier())
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Results:
print(f'Accuracy (training): {classifier.score(X_train, y_train):.2f}')
print(f'Accuracy (test): {classifier.score(X_test, y_test):.2f}', '\n')
# detailed classification report for each class:
# print(le.classes_, '\n')  # print the actual labels
print([f'{i} {cls}' for i, cls in enumerate(le.classes_)])  # print the actual labels
print('\n')
print(f'n Classification Report (test): {metrics.classification_report(y_test, y_pred)}')
