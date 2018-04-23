import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats


#bring in the six packs
#df_train = pd.read_csv('data/train.csv')
df_train = pd.read_csv('data/train_age_revised.csv')


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Missing_Total', 'Missing_Percent'])
print(missing_data.head(20))
missing_data.to_csv('output/Missing_data_summary.csv', index=False)
missing_data.to_csv('output/Missing_data_summary_with_id.csv', index=True)
