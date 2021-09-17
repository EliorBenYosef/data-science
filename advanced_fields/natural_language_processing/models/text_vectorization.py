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


class Vectorizer:

    def __init__(self, X, binary=False, ngram_range=(1,1), min_df=1, max_features=None, stopwords_removed_manually=True, analyzer='word'):
        self.X = X
        self.binary = binary
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.analyzer = analyzer
        self.stop_words = None if stopwords_removed_manually else 'english'
        self.word_vectorizer = None

    def bag_of_words(self):
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
        self.word_vectorizer = CountVectorizer(
            binary=self.binary, ngram_range=self.ngram_range, min_df=self.min_df, max_features=self.max_features,
            analyzer=self.analyzer, stop_words=self.stop_words
        )

        feature_vector = self.word_vectorizer.fit_transform(self.X).toarray()
        # word_vectorizer_analyzer = CountVectorizer().build_analyzer()
        return feature_vector

    def tf_idf(self, use_idf=True, sublinear_tf=False, norm='l2'):
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
        self.word_vectorizer = TfidfVectorizer(
            binary=self.binary, ngram_range=self.ngram_range, min_df=self.min_df, max_features=self.max_features,
            analyzer=self.analyzer, stop_words=self.stop_words,
            sublinear_tf=sublinear_tf, use_idf=use_idf, norm=norm
        )
        feature_vector = self.word_vectorizer.fit_transform(self.X).toarray()
        # word_vectorizer_analyzer = TfidfVectorizer().build_analyzer()
        return feature_vector

    def word_2_vec(self):
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
# jd_vec, word_vectorizer = tf_idf(jd_keywords, analyzer=get_stemmed_words)
# # jd_vec, word_vectorizer = bag_of_words(jd_keywords, analyzer=get_stemmed_words)