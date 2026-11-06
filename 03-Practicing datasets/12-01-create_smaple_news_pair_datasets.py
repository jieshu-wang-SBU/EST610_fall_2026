#%%
import pandas as pd
import numpy as np

date_location_matched=pd.read_csv("/Users/jieswang/PycharmProjects/news-deserts/data/010-Identifying_overlapping_content/nlp_features_computed_between_wvua_wvtm_20251209.csv",
                                  nrows=50)

#%%
date_location_matched['semantic_similarity'].describe()
#%%
sample_n = 50
strata_col = 'semantic_similarity'
date_location_matched["strata"] = pd.qcut(date_location_matched[strata_col], q=sample_n, labels=False)

stratified_sample = date_location_matched.groupby("strata", group_keys=False).apply(
    lambda x: x.sample(1, random_state=42))
#%%
stratified_sample[strata_col].describe()
#%%
save_cols = ['index_wuva', 'Title_wuva', 'Date Published_wuva',
       'Description_wuva', 'index_wvtm', 'Title_wvtm',
       'Date Published_wvtm', 'Description_wvtm',
       'wuva_tokens', 'wvtm_tokens', 'jaccard_sim',
       'tokenized_jaccard_similarity',
       'tfidf_cosine', 'semantic_similarity']

stratified_sample[save_cols].to_csv("/Users/jieswang/PycharmProjects/EST610_fall_2026/01-Lectures/Week12-Text_analysis_and_NLP/data/news_pairs.csv",
                                    index=False)
print("saved")
#%%