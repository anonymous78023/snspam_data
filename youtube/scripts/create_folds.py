"""
This script splits the data into 10 (train, val, test) folds.
"""
import os
import pandas as pd


def main():

    print('reading in data...')
    pd.set_option('display.width', 181)
    df = pd.read_csv('../processed/comments.csv')
    out_dir = '../processed/folds/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fold in range(10):
        print('creating fold %d...' % fold)
        train_df, val_df, test_df = split(df, fold)
        train_df.to_csv('%strain_%d.csv' % (out_dir, fold), index=None)
        val_df.to_csv('%sval_%d.csv' % (out_dir, fold), index=None)
        test_df.to_csv('%stest_%d.csv' % (out_dir, fold), index=None)


def split(df, fold, train_size=0.75, val_size=0.025):
    """Split the data into train, val, test depending on the fold."""

    if fold == 0:
        start, end = 0, 2000000
    elif fold == 1:
        start, end = 492386, 2492386
    elif fold == 2:
        start, end = 984772, 2984772
    elif fold == 3:
        start, end = 1477158, 3477158
    elif fold == 4:
        start, end = 1969544, 3969544
    elif fold == 5:
        start, end = 2461930, 4461930
    elif fold == 6:
        start, end = 2954316, 4954316
    elif fold == 7:
        start, end = 3446702, 5446702
    elif fold == 8:
        start, end = 3939088, 5939088
    elif fold == 9:
        start, end = 4431474, 6431474

    fold_df = df[start:end]

    split1 = int(len(fold_df) * train_size)
    split2 = split1 + int(len(fold_df) * val_size)

    train_df = fold_df[:split1]
    val_df = fold_df[split1:split2]
    test_df = fold_df[split2:]

    return train_df, val_df, test_df

if __name__ == '__main__':
    main()
