"""
(Unsupervised) Resume Screening Task via Phrase Matching.
1. Text Cleaning
2. Keywords Matching (Unsupervised)
    - search for the keywords in the resume
    - count the keywords occurrences
    - sum the keywords occurrences for each category
"""

import os
import pandas as pd
from collections import Counter
from spacy import load
# import en_core_web_sm
from spacy.matcher import PhraseMatcher
from advanced_fields.natural_language_processing.utils import TextExtractor, TextCleaner
import matplotlib.pyplot as plt
from utils import plot_hist_sum


################################

# run from terminal to download:
# python -m spacy download en_core_web_sm  # Downloads best-matching package version for the spaCy installation
# python -m spacy download en_core_web_sm-3.0.0 --direct  # Download exact package version

nlp = load('en_core_web_sm')
# nlp = en_core_web_sm.load()

################################

# keyword_df - should contain all skill sets for each category:
#   (can also be a dictionary or table)
keyword_df = pd.read_csv('../../../datasets/per_field/nlp/categories_keywords.csv')
categories = keyword_df['Category'].values
matcher = PhraseMatcher(nlp.vocab)
for i in range(len(categories)):
    cat = keyword_df['Category'].values[i]
    keywords = keyword_df['Keywords'].values[i].split('; ')
    words = [nlp(keyword) for keyword in keywords]
    matcher.add(cat, None, *words)

tc = TextCleaner(apply_normalization=False)  # stemming hinders keywords matching, i.e. 'machine learning' --> machine learn


def perform_phrase_matching(text):
    doc = nlp(text)

    matches = []
    for matcher_cat_id, start_word_i, end_word_i in matcher(doc):
        cat = nlp.vocab.strings[matcher_cat_id]  # get the unicode ID, i.e. 'COLOR'
        keyword = doc[start_word_i: end_word_i].text  # get the matched slice of the doc
        matches.append((cat, keyword))

    matches = [[cat, keyword, count] for (cat, keyword), count in Counter(matches).items()]
    matches_df = pd.DataFrame(matches, columns=['Category', 'Keyword', 'Count'])
    return matches_df


def build_candidate_profile_df(file_path):
    text = TextExtractor.read_pdf(file_path)
    text = tc.clean_resume(text)
    matches_df = perform_phrase_matching(text)

    candidate_name = tc.get_candidate_name_from_filename(file_path)
    candidate_name_df = pd.DataFrame([candidate_name], columns=['Candidate'])

    candidate_profile_df = pd.concat(
        [candidate_name_df['Candidate'], matches_df['Category'], matches_df['Keyword'], matches_df['Count']],
        axis=1)
    candidate_profile_df['Candidate'].fillna(candidate_profile_df['Candidate'].iloc[0], inplace=True)
    return candidate_profile_df


def show_accumulated_bar_per_candidate(df, title):
    ax = df.plot.barh(title=title, legend=False, figsize=(18, 7), stacked=True)

    # Add text to the bars:
    labels = []
    for cat in df.columns:  # columns first
        for candidate in df.index:  # rows second
            labels.append(f'{cat}\n{df.loc[candidate][cat]}')
    patches = ax.patches
    for label, rect in zip(labels, patches):
        width = rect.get_width()
        if width > 0:
            x = rect.get_x()
            y = rect.get_y()
            height = rect.get_height()
            ax.text(x + width / 2., y + height / 2., label, ha='center', va='center')
    plt.rcParams.update({'font.size': 10})

    plt.tight_layout()
    plt.savefig('results/category_keywords_accumulated_bar_per_candidate.png')
    plt.show()


def show_multiple_bars_per_category_per_candidate(df, title):
    categories = df_final.columns.values
    candidates = df_final.index.values
    data = df.to_numpy()

    ylabel = 'Keywords Count'
    xlabel = 'Categories'

    plot_hist_sum(data, candidates, ylabel, xlabel, title, x_tick_labels=categories)
    plt.savefig('results/category_keywords_count_histogram.png')
    plt.show()


################################


if __name__ == '__main__':

    df_initial = pd.DataFrame()
    files_dir = '../../../datasets/per_field/nlp/pdf_resumes/'
    files_paths = [os.path.join(files_dir, f) for f in os.listdir(files_dir)
                   if os.path.isfile(os.path.join(files_dir, f))]
    for file_path in files_paths:
        candidate_df = build_candidate_profile_df(file_path)
        df_initial = df_initial.append(candidate_df)

    # Count keyword occurrences under each category:
    df_final = df_initial['Keyword'].groupby([
        df_initial['Candidate'], df_initial['Category']
    ]).count().unstack(fill_value=0)  # fills nan and keeps dtype int64
    df_final.to_csv('results/candidates_profiles.csv')
    # df_final_no_i = df_final.reset_index()
    # df_final = df_final_no_i.iloc[:, 1:]
    # df_final.index = df_final_no_i['Candidate']

    # categories count sum visualization:
    title = 'Resume Keywords by Category'
    show_accumulated_bar_per_candidate(df_final, title)
    show_multiple_bars_per_category_per_candidate(df_final, title)
