import argparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from preprocess import reader, aggregate, engineer_features
import pickle
import os
from sklearn.neural_network import MLPClassifier
import pandas as pd
from xgboost.sklearn import XGBClassifier


def train_models(train_data, test_data):
    X_train, y_train = train_data.drop(['SepsisLabel_max', 'patient_id_max'], axis=1), train_data[
        'SepsisLabel_max']
    X_test, y_test = test_data.drop(['SepsisLabel_max', 'patient_id_max'], axis=1), test_data[
        'SepsisLabel_max']

    Imputer = SimpleImputer(strategy="mean", add_indicator=True)
    impute_scale_pipeline = make_pipeline(Imputer, StandardScaler())

    logistic_pipe = make_pipeline(impute_scale_pipeline, LogisticRegression(solver='lbfgs'))
    rf_pipe = make_pipeline(impute_scale_pipeline, RandomForestClassifier())
    svc_pipe = make_pipeline(impute_scale_pipeline, SVC())
    gb_pipe = make_pipeline(impute_scale_pipeline, GradientBoostingClassifier())
    dt_pipe = make_pipeline(impute_scale_pipeline, DecisionTreeClassifier())
    mlp_pipe = make_pipeline(impute_scale_pipeline, MLPClassifier())

    params = {'xgbclassifier__colsample_bytree': 1.0,
              'xgbclassifier__gamma': 1,
              'xgbclassifier__max_depth': 3,
              'xgbclassifier__min_child_weight': 1,
              'xgbclassifier__subsample': 1.0}

    XGB_pipe_scale = make_pipeline(impute_scale_pipeline, XGBClassifier(params=params))

    modeln = ['logistic_regression', 'random_forest', 'SVM', 'GradientBoost', 'DecisionTree', 'MLP', 'XGB']
    pipelines = [logistic_pipe, rf_pipe, svc_pipe, gb_pipe, dt_pipe, mlp_pipe, XGB_pipe_scale]

    path = 'trained_models'

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("new cache directory created!")

    for i, pipeline in enumerate(pipelines):
        model = pipeline.fit(X_train, y_train)
        filename = f'trained_models/{modeln[i]}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Printing metrics for : {modeln[i]} \n ")

        show_metrics(modeln[i], model, X_test, y_test)
    print("trained models saved! \n ")


def show_metrics(model_type, clf, X_test, y_test):
    predicted = clf.predict(X_test)
    f1 = f1_score(y_test, predicted, average='binary')
    acc = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted)
    recall = recall_score(y_test, predicted)
    print(f'{model_type} accuracy {acc: .3f}')
    print(f'{model_type} recall {recall: .3f}')
    print(f'{model_type} precision {precision: .3f}')
    print(f'{model_type} f1 score {f1: .3f}')
    print('\n')


def main(processed_train_file, test_folder):
    train_data = reader(processed_train_file)
    train_data = engineer_features(train_data)
    Test = reader(test_folder)
    Test = engineer_features(Test, train=False)
    print('Training models started...\n')
    train_models(train_data, Test)
    print('Training Finished\n')
    print('All models saved in cache folder\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--processed_train', type=str, help='path to processed data',
                        default='data/train')
    parser.add_argument('--test_folder', type=str, help='path to test folder',
                        default='data/test')
    args = parser.parse_args()
    main(args.processed_train, args.test_folder)
