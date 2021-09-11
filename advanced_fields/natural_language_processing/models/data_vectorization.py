"""
models for data (text) vectorization
- Bag of Words
- Tf-Idf
- Word2Vec
it's possible to use a combination of the above methods.

Note that the model can receive a 'stop_words' argument:
    stop_words='english' - uses a built-in stopword list for English. Note that there are several known issues with
        'english' and you should consider an alternative (remove stop_words manually).
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bag_of_words(X, manual_stopwords=True):
    """
    Bag of Words
        creating a sparse feature matrix containing the clean texts’ entire words count.
    """
    # # Option 1 - taking the 95% most frequent words:
    # word_vectorizer = CountVectorizer()
    # n_words = word_vectorizer.fit_transform(X).toarray().shape[1]
    # print('number of unique words:', n_words)
    # word_vectorizer = CountVectorizer(max_features=round(n_words * 0.95))

    # Option 2 - taking the words with a minimum count of 2:
    if manual_stopwords:
        word_vectorizer = CountVectorizer(min_df=2)
    else:
        word_vectorizer = CountVectorizer(min_df=2, stop_words='english')

    word_features = word_vectorizer.fit_transform(X)
    return word_features, word_vectorizer


def tf_idf(X, manual_stopwords=True):
    """
    Term Frequency - Inverse Document Frequency (TF-IDF)
    """
    if manual_stopwords:
        word_vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=1500)
    else:
        word_vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=1500, stop_words='english')
    word_features = word_vectorizer.fit_transform(X)
    return word_features, word_vectorizer


def word_2_vec():
    """
    Word2Vec
    """
    pass


