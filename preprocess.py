import argparse
import numpy as np
import pandas as pd
import os
import pickle
import regex as re


def reader(input_folder='data/train', interpolate=True):
    filesnames = os.listdir(input_folder)
    regex = re.compile(r'\d+')
    ids = [int(x) for x in regex.findall(str(filesnames))]
    dfs = list()
    for patient_id, filename in enumerate(filesnames):
        pdf = pd.read_csv(input_folder + "/" + filename, sep='|')
        sepsislabel_true_list = pdf[pdf['SepsisLabel'] == 1].index
        if not sepsislabel_true_list.empty:
            pdf = pdf[:min(pdf[pdf['SepsisLabel'] == 1].index) + 1]
            pdf['SepsisLabel'] = 1
        pdf['patient_id'] = ids[patient_id]
        if interpolate:
            pdf.interpolate(limit_direction='both', axis=0, inplace=True)  # show first hist without it
        dfs.append(pdf)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # df.to_pickle(file_name)
    return df


def aggregate(data, mean=True, std=True, aggmin=True, aggmax=True, skew=True, kurt=True,
              lastrows=False) -> pd.DataFrame:
    """
    :param data: df to be aggregated
    :param lastrows:
    :param kurt:
    :param skew:
    :param aggmax:
    :param aggmin:
    :param std:
    :param mean:
    :param patient_df: patient dataframe
    :param amount_of_rows: amount of rows to keep before the first true sepsis label
    :return: aggregated df.
    """
    df = data.copy()
    max_agg = ['Lactate', 'WBC', 'Glucose', 'Temp', 'Resp', 'Creatinine', 'BUN'] + ['Age', 'Gender', 'Unit1', 'Unit2',
                                                                                    'HospAdmTime', 'ICULOS'] + [
                  'patient_id',
                  'SepsisLabel']
    min_agg = ['SBP']

    aggdict = {}
    for col in df.columns:
        if col in min_agg:
            aggdict[col] = ['min']
        elif col in max_agg:
            aggdict[col] = ['max']
        else:
            aggdict[col] = ['mean', 'std']
    aggregated = df.groupby('patient_id').agg(aggdict)
    aggregated.columns = list(map('_'.join, aggregated.columns.values))
    return aggregated


def engineer_features(df, train=True, mean=True, std=True, aggmin=True, aggmax=True, skew=True,
                      kurt=True):
    # print('Droping Unit1, Unit2, HospAdmTime, ICULOS \n')
    # df.drop(['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'], axis=1, inplace=True)
    # df = pd.get_dummies(df, columns=['Gender'])
    if train:
        file_name = 'traintransformed.pkl'
    else:
        file_name = 'testtransformed.pkl'
    print('Aggregating...')
    df = aggregate(df, mean, std, aggmin, aggmax, skew, kurt, 10)
    # try to add something smart here
    print('Saving file..')
    df.to_pickle(file_name)
    print('saved pkl')
    return df


def preprocess(data_folder):
    print('Starting preproocessing...')
    df = reader(data_folder)
    newdf = engineer_features(df)
    file_name = f'data/{data_folder}_raw.csv'
    return file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing...')
    parser.add_argument('--folder', type=str, help='path to folder containing patients psvs',
                        default='data/train')
    args = parser.parse_args()
    preprocess(args.folder)
