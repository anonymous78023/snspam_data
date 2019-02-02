"""
This script computes features for YouTube.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz


def main(fold, limited=False):

    print('fold %d, limited: %s' % (fold, limited))

    in_dir = '../processed/folds/'
    out_dir = '../features/independent/'
    out_dir += 'limited/' if limited else 'full/'

    pd.set_option('display.width', 181)
    train_df = pd.read_csv('%strain_%d.csv' % (in_dir, fold))
    val_df = pd.read_csv('%sval_%d.csv' % (in_dir, fold))
    test_df = pd.read_csv('%stest_%d.csv' % (in_dir, fold))

    train_features_matrix, count_vectorizer = features(train_df, limited)
    val_features_matrix, _ = features(val_df, limited, count_vectorizer)
    test_features_matrix, _ = features(test_df, limited, count_vectorizer)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_npz('%strain_data_%d' % (out_dir, fold), train_features_matrix)
    save_npz('%sval_data_%d' % (out_dir, fold), val_features_matrix)
    save_npz('%stest_data_%d' % (out_dir, fold), test_features_matrix)


def features(df, limited, count_vectorizer=None):
        features_df = df.copy()

        print('building ngram features...')
        ngram_matrix, count_vectorizer = _ngrams(df, count_vectorizer=count_vectorizer)

        print('building content features...')
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        features_df['num_chs'] = df['text'].str.len()
        features_df['wday'] = df['timestamp'].dt.dayofweek
        features_df['hour'] = df['timestamp'].dt.hour
        features = ['num_chs', 'wday', 'hour', 'polarity', 'subjectivity']

        print('building sequential features...')
        features_df['len'] = features_df['text'].str.len()
        features_df['usr_msg_cnt'] = features_df.groupby('user_id').cumcount()
        features_df['usr_msg_max'] = features_df.groupby('user_id')['len'].cummax()
        features_df['usr_msg_min'] = features_df.groupby('user_id')['len'].cummin()
        features_df['usr_msg_mean'] = list(features_df.groupby('user_id')['len']
                                                      .expanding().mean().reset_index()
                                                      .sort_values('level_1')['len'])
        features += ['com_id', 'usr_msg_cnt', 'usr_msg_max', 'usr_msg_min', 'usr_msg_mean']

        features_df = features_df[features]
        features_matrix = csr_matrix(features_df.astype(float).values)
        features_matrix = hstack([features_matrix, ngram_matrix])

        return features_matrix, count_vectorizer


def _ngrams(df, count_vectorizer=None):

    df['text'] = df['text'].fillna('')
    str_list = df['text'].tolist()

    if count_vectorizer is None:
        count_vectorizer = CountVectorizer(stop_words='english', min_df=1,
                                           ngram_range=(3, 3), max_df=1.0,
                                           max_features=10000, analyzer='char_wb',
                                           binary=True, vocabulary=None, dtype=np.int32)
        ngram_matrix = count_vectorizer.fit_transform(str_list)

    else:
        ngram_matrix = count_vectorizer.transform(str_list)

    id_matrix = ss.lil_matrix((len(df), 1))
    ngram_matrix = ss.hstack([id_matrix, ngram_matrix]).tocsr()

    return ngram_matrix, count_vectorizer

if __name__ == '__main__':
    for i in range(10):
        main(fold=i, limited=False)
