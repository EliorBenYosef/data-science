"""
SOTA word embedding techniques (i.e. doc2vec)
https://towardsdatascience.com/word2vec-for-phrases-learning-embeddings-for-more-than-one-word-727b6cf723cf
"""

import os
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import remove_stopwords
import nltk


def tokenize_document(text_file):
    tokens = nltk.word_tokenize(text_file)
    return tokens


def tag_tokens(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens


def get_doc2vec_similarity(jd_file_path, resume_files_paths):
    train_corpus = list(read_corpus(resume_files_paths))
    test_corpus = list(read_corpus([jd_file_path], tokens_only=True))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=40)
    model.build_vocab(train_corpus)
    # print(model.wv.vocab)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    inferred_vector = model.infer_vector(test_corpus[0])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    # print(sims)
    return sims


def read_corpus(doc_corpus, tokens_only=False):
    for i in range(len(doc_corpus)):
        tokens = gensim.utils.simple_preprocess(doc_corpus[i])
        selected_tokens = []
        for token in tag_tokens(gensim.utils.simple_preprocess(doc_corpus[i])):
            if token[1][0] in ('N', 'V', 'F'):
                selected_tokens.append(token[0])
            else:
                selected_tokens.append(token[0])
        if tokens_only:
            yield selected_tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(selected_tokens, [i])


def read_corpus_and_lemmatize(doc_corpus, tokens_only=False):
    doc_tokens = []
    for i in range(len(doc_corpus)):
        tokens = []
        temp_sentence = remove_stopwords(doc_corpus[i])
        # tokens = gensim.utils.lemmatize(temp_sentence,allowed_tags=re.compile('(NN|VB|)'))
        for token in tag_tokens(tokenize_document(temp_sentence)):
            if token[1][0] in ('N', 'V'):
                print(token[0])
                print(token[1])
                tokens.append(token[0])
        if tokens_only:
            doc_tokens.append(tokens)
        else:
            doc_tokens.append(TaggedDocument(words=tokens, tags=[i]))
    return doc_tokens


def process_files(jd_file_path, resume_files_paths):
    final_doc_rating_list = []
    sim = get_doc2vec_similarity(jd_file_path, resume_files_paths)
    for element in sim:
        doc_rating_list = []
        doc_rating_list.append(os.path.basename(resume_files_paths[element[0]]))
        doc_rating_list.append("{:.0%}".format(element[1]))
        final_doc_rating_list.append(doc_rating_list)
    return final_doc_rating_list


if __name__ == "__main__":
    jd_file_path = 'D:\\learning\\data\\Data\\JD\\WalletShare - P2 ETL DataStage JD.docx'
    resume_files_paths = ['D:\\learning\\data\\Data\\Resume\\Srinivas Sivadasu.docx',
                          'D:\\learning\\data\\Data\\Resume\\Pavan Reddy.docx',
                          'D:\\learning\\data\\Data\\Resume\\Veeranjaneyulu Tokala.docx',
                          'D:\\learning\\data\\Data\\Resume\\Sana Reddy.docx',
                          'D:\\learning\\data\\Data\\Resume\\Vishwanath A.docx',
                          'D:\\learning\\data\\Data\\Resume\\Udaya Bhaskar.docx',
                          'D:\\learning\\data\\Data\\Resume\\Ravichandra Reddy.docx',
                          'D:\\learning\\data\\Data\\Resume\\Ravi Kumar.docx',
                          'D:\\learning\\data\\Data\\Resume\\Ambati Nageswararao .docx']
    process_files(jd_file_path, resume_files_paths)
