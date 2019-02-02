"""
Script that connects messages by text similarity.
"""
import os
import pandas as pd


def main(fold):

    # settings
    in_dir = '../processed/folds/'
    out_dir = '../features/relational/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pd.set_option('display.width', 181)
    print('fold %d' % fold)

    # concatentate messags for one fold
    print('reading train, val, and test sets..')
    train_df = pd.read_csv(in_dir + 'train_%d.csv' % fold)
    val_df = pd.read_csv(in_dir + 'val_%d.csv' % fold)
    test_df = pd.read_csv(in_dir + 'test_%d.csv' % fold)

    print('concatenating train, val, and test sets..')
    df = pd.concat([train_df, val_df, test_df])
    df = df[['com_id', 'text']]

    # assign each text an text id
    print('assigning each text an ID...')
    df['text'] = df['text'].fillna('')
    gf = df.groupby('text').size().reset_index()
    gf = gf[gf[0] > 1]
    gf = gf[gf['text'] != '']
    gf['text_id'] = range(len(gf))
    del gf[0]

    # merge messages and text ids
    print('merging text IDs with messages...')
    rf = df.merge(gf, on='text', how='left')
    rf = rf[~pd.isnull(rf['text_id'])]
    rf.to_csv('%stext_relation_%d.csv' % (out_dir, fold), index=None)

if __name__ == '__main__':
    for i in range(10):
        main(fold=i)
