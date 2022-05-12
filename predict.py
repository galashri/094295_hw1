import argparse
import os
import pandas as pd

# read file as pandas dataframe
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

dataframe = pd.read_csv("data/train/patient_0.psv", sep='|')
print(dataframe)