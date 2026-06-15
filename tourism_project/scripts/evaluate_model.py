import os, pickle
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, roc_auc_score

HF_TOKEN        = os.environ['HF_TOKEN']
HF_USERNAME     = os.environ['HF_USERNAME']
DATASET_REPO_ID = f'{HF_USERNAME}/visit-with-us-tourism'
MODEL_REPO_ID   = f'{HF_USERNAME}/visit-with-us-wellness-model'

te = load_dataset(DATASET_REPO_ID, data_files='data/test.csv', split='train', token=HF_TOKEN).to_pandas()
X_te, y_te = te.drop('ProdTaken', axis=1), te['ProdTaken']

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename='model.pkl', token=HF_TOKEN)
with open(model_path, 'rb') as f: model = pickle.load(f)

y_pred = model.predict(X_te)
y_prob = model.predict_proba(X_te)[:, 1]
print('=== Model Evaluation ===')
print(classification_report(y_te, y_pred, target_names=['No Purchase', 'Purchase']))
print(f'ROC-AUC: {roc_auc_score(y_te, y_prob):.4f}')
