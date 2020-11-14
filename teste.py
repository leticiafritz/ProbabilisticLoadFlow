import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='IPAGothic')
import numpy as np
import statsmodels.api as sm

train = pd.read_csv('../input/train.csv', parse_dates=['date'], index_col='date')
test = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col='date')
df = pd.concat([train, test], sort=True)
sample = pd.read_csv('../input/sample_submission.csv')
