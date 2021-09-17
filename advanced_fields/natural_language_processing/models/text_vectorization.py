"""
Steps:
1. Text Pre-Processing
2. Text Vectorization - converting text to vector form.
3. Applying either:
    - ML/DL model on the vectors
    - similarity measure between 2 vectors

Text Vectorization models/methods:
- Bag of Words (BoW)
- Term Frequency - Inverse Document Frequency (TF-IDF)
- Word2Vec (W2V)
it's possible to use a combination of the above methods.

data you can get from word_vectorizer:
print('Stop words:' ,word_vectorizer.get_stop_words())
print('Features:', word_vectorizer.get_feature_names())

Note that the model can receive a 'stop_words' argument:
    stop_words='english' - uses a built-in stopword list for English. Note that there are several known issues with
        'english' and you should consider an alternative (remove stop_words manually).
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer


def bag_of_words(X, min_df=1, binary=False, stopwords_removed_manually=True, analyzer='word'):
    """
    Bag of Words (BoW) - counts words occurrences.
        creates a sparse feature matrix containing the clean texts’ entire words count.
    Binary BoW - considers words' existence (0\1), no .matter how many occurrences
    """
    # # Option 1 - taking the 95% most frequent words:
    # word_vectorizer = CountVectorizer()
    # n_words = word_vectorizer.fit_transform(X).toarray().shape[1]
    # print('number of unique words:', n_words)
    # word_vectorizer = CountVectorizer(max_features=round(n_words * 0.95))

    # Option 2 - taking the words with a minimum count of 2:
    word_vectorizer = CountVectorizer(min_df=min_df, binary=binary, analyzer=analyzer,
                                      stop_words=None if stopwords_removed_manually else 'english')

    feature_vector = word_vectorizer.fit_transform(X).toarray()
    # word_vectorizer_analyzer = CountVectorizer().build_analyzer()
    return feature_vector, word_vectorizer


def tf_idf(X, max_features=None, use_idf=True, binary=False, sublinear_tf=False, stopwords_removed_manually=True, analyzer='word'):
    """
    Term Frequency - Inverse Document Frequency (TF-IDF)
    Binary TF-IDF - regarding the tf term only

    The formula of TF-IDF for a term t of a document d in n documents is:
    TF-IDF(t, d) = TF(t, d) * IDF(t)
    TF(t, d) = # t in d / # T in d
        sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    IDF(t) = log_e [# D / # d's with t]
            DF(t) = # d with t --> the document frequency of t
        IDF measures how important a word is.
        if a word is present in all the docs --> # d's with t == # D --> log(1) = 0 --> no importance.
            to make sure that terms with IDF==0 will not be entirely ignored, we use:
                IDF(t) + 1 (if smooth_idf=False)
                IDF(t) = log_e [ n / df(t) ] + 1
            this differs from the standard textbook notation that defines the idf as:
                IDF(t) = log_e [ n / (df(t) + 1) ]
        if a word is present in a single doc --> # d's with t == 1 --> log(# docs) > 0 --> highest importance.

    """
    word_vectorizer = TfidfVectorizer(sublinear_tf=sublinear_tf, max_features=max_features,
                                      use_idf=use_idf, binary=binary, analyzer=analyzer, norm='l2',
                                      stop_words=None if stopwords_removed_manually else 'english')
    feature_vector = word_vectorizer.fit_transform(X).toarray()
    # word_vectorizer_analyzer = TfidfVectorizer().build_analyzer()
    return feature_vector, word_vectorizer


def word_2_vec():
    """
    Word2Vec
    """
    # TODO: complete
    pass



# wnl = WordNetLemmatizer()
# word_vectorizer_analyzer = TfidfVectorizer().build_analyzer()
# word_vectorizer_analyzer = CountVectorizer().build_analyzer()
#
# def get_stemmed_words(doc):
#     return (wnl.lemmatize(word) for word in word_vectorizer_analyzer(doc) if word not in set(stopwords.words('english')))
#
# # jd_vec = req_vector:
# jd_vec, word_vectorizer = tf_idf(jd_keywords, analyzer=get_stemmed_words)
# # jd_vec, word_vectorizer = bag_of_words(jd_keywords, analyzer=get_stemmed_words)