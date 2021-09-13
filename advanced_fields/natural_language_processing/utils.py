import re
# from nltk import download
# download('stopwords')
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from unidecode import unidecode


class TextCleaner:

    # re.sub(r"\s+", "", text)  # remove all spaces
    # re.sub(r"^\s+", "", text)  # remove left (leading) space
    # re.sub(r"\s+$", "", text)  # remove right (trailing) space
    # re.sub(r"^\s+|\s+$", "", text)  # remove both
    # re.sub(' +', ' ',text)  # replace multiple spaces with a single space

    def __init__(self, remove_stopwords=True, apply_stemming=True):
        """
        stopwords - irrelevant words to exclude when cleaning the texts:
            articles (the, this, a, an, ...)
            conjunctions (and, but, or, while, because, ...)
            pronouns (I, you, ...)
            prepositions (by, with, about, until, ...)
            'not'
            ...
        http://www.butte.edu/departments/cas/tipsheets/grammar/parts_of_speech.html

        stemming - taking only the root of the word.
        """
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self.stop_words = stopwords.words('english')  # will be removed from the text

        self.apply_stemming = apply_stemming
        if self.apply_stemming:
            self.ps = PorterStemmer()

    def remove_stopwords_f(self, text):
        text = [word for word in text.split() if word not in set(self.stop_words)]
        text = ' '.join(text)  # convert list back to string
        return text

    def apply_stemming_f(self, text):
        text = [self.ps.stem(word) for word in text.split()]
        text = ' '.join(text)  # convert list back to string
        return text

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

        if self.remove_stopwords and self.apply_stemming:
            text = self.remove_stopwords_and_apply_stemming(text)
        else:
            if self.remove_stopwords:
                text = self.remove_stopwords_f(text)
            if self.apply_stemming:
                text = self.apply_stemming_f(text)

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

        if self.remove_stopwords and self.apply_stemming:
            text = self.remove_stopwords_and_apply_stemming(text)
        else:
            if self.remove_stopwords:
                text = self.remove_stopwords_f(text)
            if self.apply_stemming:
                text = self.apply_stemming_f(text)

        return text

    @staticmethod
    def clean_resume_file_name(text):
        text = text.lower()
        text = text.replace('resume', '').replace('cv', '')
        text = text.replace('-', ' ').replace('_', ' ')
        text = text.replace('english', '').replace('eng', '')
        text = re.sub('[^a-zA-Z ]', repl='', string=text)  # remove non-letters characters (i.e. punctuation)
        text = re.sub(' +', ' ', text)  # remove extra whitespace
        text = re.sub(r"^\s+|\s+$", "", text)  # remove both
        return text


def read_pdf(file_path):
    file_reader = PyPDF2.PdfFileReader(open(file_path, 'rb'))  # rb - read binary
    text = []
    for i in range(file_reader.getNumPages()):
        text.append(file_reader.getPage(i).extractText())
    text = ' '.join(text)  # convert list back to string
    text = text.replace('\n', '')
    return text