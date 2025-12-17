pip install mysql.connector

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Banking (1).csv')

data.head()

data.shape
data.info()
data.describe()
data.isnull().sum()
data.columns

bins = [0, 100000, 200000, 300000, 400000, float('inf')]
labels = ['0 - 100K', '101K - 200K', '201K - 300K', '301K - 400K', '400K+']

data['Inacome Band'] = pd.cut(data["Estimated Income"], bins = bins, labels = labels, right = False)
data['Inacome Band'].head()

data['Inacome Band'].value_counts().plot(kind = 'bar')

categorical_cols = ['BRId', 'GenderId', 'IAId', 'Nationality', 'Occupation', 'Fee Structure',
       'Loyalty Classification', 'Properties Owned',
       'Risk Weighting', 'Inacome Band']

for col in categorical_cols:
    print(f"Value counts for column ${col} : ${data[col].value_counts()}")

sns.set_theme(style="whitegrid", palette="Set2")
cat_df = data[categorical_cols]
for i, predictor in enumerate(cat_df.columns):
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=cat_df,
        x=predictor)
    plt.title(f'Distribution of {predictor}', fontsize=14, fontweight='bold')
    plt.xlabel(predictor, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

data['Occupation'].nunique()
numerical_cols = ['Age', 'Estimated Income', 'Superannuation Savings', 'Credit Card Balance', 'Bank Loans', 'Bank Deposits',
                 'Checking Accounts', 'Saving Accounts', 'Business Lending']
import matplotlib.ticker as ticker
sns.set_theme(style="whitegrid", palette="Set2")
num_df = data[numerical_cols]
for i, predictor in enumerate(num_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=num_df,
        x=predictor)
    plt.title(f'Distribution of {predictor}', fontsize=14, fontweight='bold')
    plt.xlabel(predictor, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    plt.show()
