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
from advanced_fields.natural_language_processing.models.text_vectorization import tf_idf, bag_of_words


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

        # get candidate's name
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        candidate_name = tc.clean_resume_file_name(file_name)

        candidate_df = pd.DataFrame([[candidate_name, resume_text, resume_text_clean]],
                                    columns=['Candidate', 'Resume', 'Resume_Clean'])
        df = df.append(candidate_df)

    jd_vec, word_vectorizer = tf_idf(jd_keywords)  # bag_of_words(jd_keywords)

    resumes_vec = word_vectorizer.transform(df['Resume Clean'].values).toarray()

    sm = SimilarityMetrics(v_base=jd_vec, v_compare=resumes_vec)
    sim = sm.cos_sim()  # similarity_list

    # final_doc_rating_list = []
    # sorted_doc_list = sorted(zip(sim, files_paths), key=lambda x: x[0], reverse=True)
    # for element in sorted_doc_list:
    #     doc_rating_list = []
    #     doc_rating_list.append(os.path.basename(element[1]))
    #     doc_rating_list.append("{:.0%}".format(element[0]))
    #     final_doc_rating_list.append(doc_rating_list)
    #
    # print(final_doc_rating_list)
