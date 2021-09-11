import re
# from nltk import download
# download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from unidecode import unidecode


class TextCleaner:

    def __init__(self, manual_stopwords=True):
        """
        stopwords - irrelevant words to exclude when cleaning the texts:
            articles (the, this, a, an, ...)
            conjunctions (and, but, or, while, because, ...)
            pronouns (I, you, ...)
            prepositions (by, with, about, until, ...)
            'not'
            ...

        http://www.butte.edu/departments/cas/tipsheets/grammar/parts_of_speech.html
        """
        self.manual_stopwords = manual_stopwords
        if self.manual_stopwords:
            self.stop_words = stopwords.words('english')  # will be removed from the text
            self.ps = PorterStemmer()  # used for stemming (taking only the root of the word)

    def remove_stopwords_and_apply_stemming(self, text):
        # remove stopwords & apply stemming (taking only the root of the word):
        text = [self.ps.stem(word) for word in text.split() if word not in set(self.stop_words)]
        text = ' '.join(text)  # convert list back to string
        return text

    def clean_review(self, text):
        """
        Cleans a single review (simplifies it as much as possible)
        """
        text = text.replace("n't", ' not')  # break negative contractions

        text = re.sub('[^a-zA-Z ]', repl='', string=text)  # remove non-letters characters (i.e. punctuation)

        text = re.sub(' +', ' ', text)  # remove extra whitespace
        text = text.lower()  # lowercase capital letters

        if self.manual_stopwords:
            text = self.remove_stopwords_and_apply_stemming(text)

        return text

    def clean_resume(self, text):
        """
        Cleans a single resume (resume text)
        """
        text = re.sub('httpS+s*', '', text)  # remove URLs
        text = re.sub('RT|cc', '', text)  # remove RT and cc
        text = re.sub('#S+', '', text)  # remove hashtags
        text = re.sub('@S+', '', text)  # remove mentions

        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), '', text)  # remove punctuations

        text = re.sub(r'[^\x00-\x7f]', '', text)  # remove non-ASCII characters
        # # Replace non-ASCII characters with their most alike representation (doesn't work):
        # text = unidecode(unicode(text, encoding="utf-8"))

        text = re.sub(' +', ' ', text)  # remove extra whitespace
        text = text.lower()  # lowercase capital letters

        if self.manual_stopwords:
            text = self.remove_stopwords_and_apply_stemming(text)

        return text
