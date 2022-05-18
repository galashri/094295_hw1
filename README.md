# Task 1 - Prediction of Sepsis

This repo contains Data Analysis and Prediction processes of Sepsis classification task. <br> 
To run the code use the next command in the terminal:

conda env create -f environment.yml

then run this command:

conda activate hw1_env

## Prediction

To predict with our best XGBoost model run predict.py.


## Training

To train our advanced models run the Trainer.py file. it will create "trained_models" folder and the models will be saved in it

To predict with one of the models, take the name of the saved model and use the command:

python predict.py --model_Type "logistic_regression"

Trained Models list - logistic_regression', 'random_forest', 'SVM', 'GradientBoost', 'DecisionTree', 'MLP', 'XGB'.

 
## Exploration and Data Analysis
Find under the notebooks folder 4 jupyter notebooks:
<li> Data-Analysis </li>
<li> XGBoost - containing the training process and feature engineering of XGboost model </li>
<li> Logistic Regression - containing the training process and feature engineering of LR model </li>
<li> Advanced Models Training - containing the training process and feature engineering of other model </li>
