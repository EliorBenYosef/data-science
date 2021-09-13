"""
Resume Screening Task via Text Vectorization (supervised).
Classifying resumes into job categories.

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
from advanced_fields.natural_language_processing.models.data_vectorization import tf_idf

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_csv('../../../datasets/per_field/nlp/UpdatedResumeDataSet.csv', encoding='utf-8')
cat_var = 'Category'
text_var = 'Resume'

###################################

# # 0. Exploratory Data Analysis:
# analyze_df(resumeDataSet)
# analyze_cat_var(resumeDataSet, cat_var, horizontal=False, show=False)

###################################

# 1. Data Pre-Processing

# Encoding categorical data
#   Encoding the cat_var column using LabelEncoding
#       (since it's our target label column, even though it contains ‘Nominal’ data)
#   each category will become a class and we will be building a multiclass classification model.
le = LabelEncoder()
df[cat_var] = le.fit_transform(df[cat_var])

remove_stopwords_manually = True

# Data Cleaning (cleaning the texts):
tc = TextCleaner(remove_stopwords_manually)
df[text_var + '_clean'] = df[text_var].apply(lambda x: tc.clean_resume(x))

# Data Vectorization:
X = df[text_var + '_clean'].values
word_features, word_vectorizer = tf_idf(X, remove_stopwords_manually)
X = word_features.toarray()

y = df[cat_var].values

##############################

# 2. Dataset Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##############################

# 3. Multiclass Classification:
#   building the model using: ‘One vs Rest’ method using a ‘KNeighborsClassifier’ model as an estimator.

classifier = OneVsRestClassifier(estimator=KNeighborsClassifier())
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Results:
print(f'Accuracy (training): {classifier.score(X_train, y_train):.2f}')
print(f'Accuracy (test): {classifier.score(X_test, y_test):.2f}', '\n')
# detailed classification report for each class:
print(le.classes_)  # print the actual labels
print(f'n Classification Report (test): {metrics.classification_report(y_test, y_pred)}')
