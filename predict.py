from sklearn.ensemble import RandomForestClassifier
import argparse
import pandas as pd
import pickle
import numpy as np
import os
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


def aggregate(data, mean=True, std=True, aggmin=True, aggmax=True, skew=True, kurt=True, lastrows=False)-> pd.DataFrame:
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
    #print('Droping Unit1, Unit2, HospAdmTime, ICULOS \n')
    #df.drop(['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'], axis=1, inplace=True)
    #df = pd.get_dummies(df, columns=['Gender'])
    if train:
        file_name = 'traintransformed.pkl'
    else:
        file_name = 'testtransformed.pkl'
    print('Aggregating...')
    df = aggregate(df, mean, std, aggmin, aggmax, skew, kurt)
    # try to add something smart here
    print('Saving file..')
    df.to_pickle(file_name)
    print('saved pkl')
    return df


def main(test_folder, trained_models, model_Type):
    filename = str(trained_models) + str('/') + str(model_Type) + '.pkl'
    print(filename)
    test = reader(test_folder)
    test = engineer_features(test, train=False)
    ids = test['patient_id_max']
    X_test, y_test = test.drop(['SepsisLabel_max', 'patient_id_max'], axis=1), test['SepsisLabel_max']
    model = pickle.load(open(filename, 'rb'))
    answers = {'Id': ids, 'SepsisLabel': model.predict(X_test).astype(int)}
    pd.DataFrame(answers)[['Id', 'SepsisLabel']].to_csv('prediction.csv', index=False, header=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction Arguments')
    parser.add_argument('--test_folder', type=str, help='path to test data',
                        default='data/test')
    parser.add_argument('--trained_models', type=str, help='path to trained models',
                        default='trained_models')
    parser.add_argument('--model_Type', type=str,
                        help="model type: logistic_regression', 'random_forest', 'SVM', 'GradientBoost', "
                             "'DecisionTree', 'MLP', 'XGB'",
                        default='XGB')
    args = parser.parse_args()
    main(args.test_folder, args.trained_models, args.model_Type)
