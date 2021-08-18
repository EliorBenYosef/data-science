import re
# from nltk import download
# download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def get_clean_corpus(texts, print_corpus=False):
    """
    Cleans (simplifies) the texts as much as possible

    stopwords - irrelevant words to exclude when cleaning the texts:
        articles (the, this, a, an, ...)
        conjunctions (and, but, or, while, because, ...)
        pronouns (I, you, ...)
        prepositions (by, with, about, until, ...)
        ...

    http://www.butte.edu/departments/cas/tipsheets/grammar/parts_of_speech.html
    """
    stop_words = stopwords.words('english')
    stop_words.remove('not')

    ps = PorterStemmer()

    corpus = []  # a list contains the clean texts
    for text in texts:
        # break negative contractions:
        text = text.replace("n't", ' not')
        # remove non-letters characters (i.e. punctuation):
        text = re.sub('[^a-zA-Z ]', repl='', string=text)
        # lowercase capital letters:
        text = text.lower()
        # remove stopwords & apply stemming (taking only the root of the word):
        text = [ps.stem(word) for word in text.split() if word not in set(stop_words)]
        # convert back to string:
        text = ' '.join(text)
        corpus.append(text)

    if print_corpus:
        print(corpus)

    return corpus
