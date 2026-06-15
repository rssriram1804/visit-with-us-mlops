import os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from huggingface_hub import HfApi

HF_TOKEN        = os.environ['HF_TOKEN']
HF_USERNAME     = os.environ['HF_USERNAME']
DATASET_REPO_ID = f'{HF_USERNAME}/visit-with-us-tourism'
api = HfApi()

# Load raw data from HF
ds = load_dataset(DATASET_REPO_ID, data_files='data/tourism.csv', split='train', token=HF_TOKEN)
df = ds.to_pandas()

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True, errors='ignore')

# Fix categorical anomalies
df['Gender']        = df['Gender'].replace({'Fe Male': 'Female'})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# Handle missing values
for col in df.select_dtypes('number').columns:
    if col != 'ProdTaken': df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes('object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)

# Feature engineering
df['IncomePerPerson']     = (df['MonthlyIncome'] / df['NumberOfPersonVisiting']).round(2)
df['PitchEffectiveness']  = (df['PitchSatisfactionScore'] * df['NumberOfFollowups']).round(2)
df['FamilySize']          = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df['IsFrequentTraveller'] = (df['NumberOfTrips'] > df['NumberOfTrips'].median()).astype(int)

# Encode and split
X, y = df.drop('ProdTaken', axis=1), df['ProdTaken']
le   = LabelEncoder()
for col in X.select_dtypes('object').columns:
    X[col] = le.fit_transform(X[col])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pd.concat([X_tr, y_tr], axis=1).to_csv('/tmp/train.csv', index=False)
pd.concat([X_te, y_te], axis=1).to_csv('/tmp/test.csv',  index=False)

# Upload to HF
for path, repo_path in [('/tmp/train.csv', 'data/train.csv'), ('/tmp/test.csv', 'data/test.csv')]:
    api.upload_file(path_or_fileobj=path, path_in_repo=repo_path,
                    repo_id=DATASET_REPO_ID, repo_type='dataset', token=HF_TOKEN)
print('Data preparation complete.')
