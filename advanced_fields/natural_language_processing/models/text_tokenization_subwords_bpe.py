"""
Byte Pair Encoding (BPE)

One of the most popular Subword Tokenization algorithm, widely used among transformer-based models.
merges the most frequently occurring character or character sequences iteratively.
"""

import numpy as np
import pandas as pd
from csv import QUOTE_NONE
import collections
import re


def clean_text(text):
    text = text.lower()  # lowercase capital letters

    text = re.sub('[^a-zA-Z\']+', ' ', text)  # select only alphabet characters (letters only)
    # text = re.sub('[^a-zA-Z0-9]+', ' ', text)  # select only alphanumeric characters (letters & numbers)
    # text = re.sub(r'\W+', ' ', text)  # Select only alphanumeric characters (including greek & underscore)

    text = re.sub(' +', ' ', text)  # remove extra spaces
    return text


def get_pair_frequency(corpus):
    """
    Computes the bigrams frequency in the corpus.
    bigrams = pairs of characters / character sequences.
    """
    pairs = collections.defaultdict(int)
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_pair_in_corpus(pair, corpus):
    corpus_merged = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')  # neg. lookbehind assertion + bigram + neg. lookahead assertion
    for word in corpus:
        word_merged = p.sub(''.join(pair), word)
        corpus_merged[word_merged] = corpus[word]
    return corpus_merged


def update_vocab():
    if not frequency_vocab:
        frequency_vocab[pair_merged] = frequency
        return

    pair_merged_leftover = pair_merged
    for pair in frequency_vocab.keys():
        if pair in pair_merged_leftover:
            pair_merged_leftover = pair_merged_leftover.replace(pair, '')
            if not pair_merged_leftover:
                return
    if len(pair_merged_leftover) > 1:
        frequency_vocab[pair_merged_leftover] = frequency
        return

    for pair in frequency_vocab.keys():
        if pair in pair_merged and frequency >= frequency_vocab[pair]:
            del frequency_vocab[pair]
            frequency_vocab[pair_merged] = frequency
            return

    frequency_vocab[pair_merged] = frequency


def perform_bpe(df, text_var, perform_traditional_bpe, min_frequency):
    # Text Cleaning:
    df[text_var + '_clean'] = df[text_var].apply(lambda x: clean_text(x))
    corpus_raw = df[text_var + '_clean'].values

    ##############################

    corpus_words = []
    for text in corpus_raw:
        words = text.split()  # Word Tokenization
        corpus_words.extend(words)

    # corpus_words = np.array(['low']*5 + ['lower']*2 + ['widest']*3 + ['newest']*5 + ['wider']*1)
    # # corpus_words = np.array(['low']*5 + ['lower']*2 + ['widest']*3 + ['newest']*5)
    # min_frequency = 1

    corpus_words_in_chars = [' '.join(word) for word in corpus_words]  # Character Tokenization
    corpus = dict(
        collections.Counter(corpus_words_in_chars))  # create a dict containing the frequency of each word in the corpus
    print("Corpus:", corpus)

    ##############################

    # Initialize the vocabulary with unique characters in the corpus
    if perform_traditional_bpe:
        vocab = list(set(' '.join(corpus_words)))
        vocab.remove(' ')
    else:
        global frequency_vocab
        frequency_vocab = {}

    global pair_merged, frequency, merges
    merges = []

    # num_merges = 100
    # for i in range(num_merges):
    while True:
        # perform_bpe_iteration

        pairs_frequency = get_pair_frequency(corpus)
        if not pairs_frequency:
            print('BPE Completed', '\n')
            break
        most_frequent_pair = max(pairs_frequency, key=pairs_frequency.get)
        pair_merged = ''.join(list(most_frequent_pair))  # convert the pair tuple to a string
        frequency = pairs_frequency[most_frequent_pair]
        print(f'Most Frequent pair: {most_frequent_pair}. Frequency: {frequency}')
        if frequency < min_frequency:
            print('BPE Completed', '\n')
            break

        # merge the most frequent pair in corpus
        corpus = merge_pair_in_corpus(most_frequent_pair, corpus)
        print('Corpus:', corpus)

        merges.append(most_frequent_pair)
        if perform_traditional_bpe:
            vocab.append(pair_merged)
        else:
            update_vocab()

    print('BPE Merge Operations:', merges, '\n')
    if perform_traditional_bpe:
        print('Vocabulary:', vocab, '\n')
    else:
        print('Vocabulary:', frequency_vocab, '\n')


##############################

# Applying BPE to an OOV word
# segmenting the OOV word into subwords using learned operations.


def get_subwords_bpe(word):
    subwords = ' '.join(word)  # Character Tokenization
    subwords = {subwords: 1}  # create a dictionary
    while True:
        pairs_frequency = get_pair_frequency(subwords)
        pairs_indices = [merges.index(pair) for pair in pairs_frequency.keys() if pair in merges]
        if len(pairs_indices) == 0:
            print('BPE Completed', '\n')
            break
        most_frequent_pair = merges[min(pairs_indices)]  # choose the most frequent learned operation
        subwords = merge_pair_in_corpus(most_frequent_pair, corpus=subwords)  # merge the most frequent pair in the OOV word
        print('subwords:', list(subwords.keys())[0])
    subwords = [key for key in subwords.keys()][0]
    return subwords


def get_subwords_backwards(word):
    word_leftovers = word
    subwords = []
    while word_leftovers:
        if word_leftovers in frequency_vocab.keys():
            subwords.append(word_leftovers)
            break
        for i in range(1, len(word_leftovers)):  # going backwards (gets the biggest combinations)
            subword = word_leftovers[:-i]
            if subword in frequency_vocab.keys():
                subwords.append(subword)
                word_leftovers = word_leftovers.replace(subword, '')
                break
        else:
            subwords.append(word_leftovers[0])
            word_leftovers = word_leftovers[1:]
    subwords = ' '.join(subwords)
    return subwords


def perform_bpe_and_apply_to_oov_words(df, text_var, oov_words, perform_traditional_bpe=False, min_frequency=3):
    perform_bpe(df, text_var, perform_traditional_bpe, min_frequency)

    # oov_words = ['lowest']
    for oov in oov_words:
        if perform_traditional_bpe:
            subwords = get_subwords_bpe(oov)
        else:
            subwords = get_subwords_backwards(oov)

        subwords = re.sub(' ', ' + ', subwords)
        print(f'OOV word: {oov} = {subwords}')


if __name__ == '__main__':
    df = pd.read_csv('../../../datasets/per_field/nlp/labeled_reviews.tsv', delimiter='\t', quoting=QUOTE_NONE)
    text_var = 'Review'
    oov_words = ['disrespectful', 'distasteful', 'extraordinarily', 'antinationalist', 'unhappiness']
    perform_bpe_and_apply_to_oov_words(df, text_var, oov_words, perform_traditional_bpe=False, min_frequency=3)

    # output:
    # OOV word: disrespectful = dis + respect + ful
    # OOV word: distasteful = dis + taste + ful
    # OOV word: extraordinarily = extra + or + din + ar + il + y
    # OOV word: antinationalist = ant + in + ation + al + ist
    # OOV word: unhappiness = un + ha + pp + ine + ss