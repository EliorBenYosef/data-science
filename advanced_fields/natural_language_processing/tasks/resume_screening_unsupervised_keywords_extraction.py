"""
(Unsupervised) Resume Screening Task via Keywords Extraction.
1. Keywords Extraction (Unsupervised) - from a job description (docx)
2. Keywords Expansion (#TODO)
3. Text Extraction & Cleaning - from resumes (pdf)
4. Text Vectorization - fitting a vectorizer on the job description keywords,
   and transforming both the job description keywords and the resumes texts (into the JD keywords vector).
5. Calculating Similarity - between the job description keywords vector, and each of the resumes keywords vector
"""

import os
import pandas as pd
from advanced_fields.natural_language_processing.utils import TextExtractor, TextCleaner, SimilarityMetrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

tc = TextCleaner()


if __name__ == '__main__':
    jd_text = TextExtractor.read_docx('../../../datasets/per_field/nlp/job_description.docx')
    jd_keywords = tc.extract_keywords(jd_text)

    df = pd.DataFrame()  # doc_corpus, resume_doc_text, resumes_df
    files_dir = '../../../datasets/per_field/nlp/pdf_resumes/'
    files_paths = [os.path.join(files_dir, f) for f in os.listdir(files_dir)
                   if os.path.isfile(os.path.join(files_dir, f))]
    for file_path in files_paths:
        resume_text = TextExtractor.read_pdf(file_path)
        resume_text_clean = tc.clean_resume(resume_text)

        candidate_name = tc.get_candidate_name_from_filename(file_path)
        candidate_df = pd.DataFrame([[candidate_name, resume_text, resume_text_clean]],
                                    columns=['Candidate', 'Resume', 'Resume_Clean'])
        df = df.append(candidate_df)

    word_vectorizer = TfidfVectorizer(vocabulary=jd_keywords)
    # print('Features:', word_vectorizer.get_feature_names())

    jd_keywords_text = np.expand_dims(np.array(' '.join(jd_keywords)), axis=0)
    jd_vec = word_vectorizer.fit_transform(jd_keywords_text).toarray()
    resumes_vec = word_vectorizer.transform(df['Resume_Clean'].values).toarray()

    sm = SimilarityMetrics(v_base=jd_vec, v_compare=resumes_vec)
    similarity_list = sm.cos_sim()
    df.insert(len(df.columns), 'Similarity', similarity_list)
    df.sort_values(['Similarity'], ascending=False, inplace=True, kind='quicksort')
    df.insert(len(df.columns), 'JD Match %', df['Similarity'].map(lambda x: f'{x:.0%}'))

    print(df[['Candidate', 'JD Match %']].to_string(index=False))
