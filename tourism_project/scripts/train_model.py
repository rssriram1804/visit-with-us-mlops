import os, pickle, mlflow, mlflow.sklearn, pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

HF_TOKEN        = os.environ['HF_TOKEN']
HF_USERNAME     = os.environ['HF_USERNAME']
DATASET_REPO_ID = f'{HF_USERNAME}/visit-with-us-tourism'
MODEL_REPO_ID   = f'{HF_USERNAME}/visit-with-us-wellness-model'
api = HfApi()

tr = load_dataset(DATASET_REPO_ID, data_files='data/train.csv', split='train', token=HF_TOKEN).to_pandas()
te = load_dataset(DATASET_REPO_ID, data_files='data/test.csv',  split='train', token=HF_TOKEN).to_pandas()
X_tr, y_tr = tr.drop('ProdTaken', axis=1), tr['ProdTaken']
X_te, y_te = te.drop('ProdTaken', axis=1), te['ProdTaken']

mlflow.set_experiment('Visit_With_Us_WellnessTourism')
best_score, best_model_obj = 0, None

MODELS = {
    'DecisionTree':     (DecisionTreeClassifier(random_state=42),
                         {'max_depth': [3,5,10,None], 'min_samples_split': [2,5,10], 'criterion': ['gini','entropy']}),
    'Bagging':          (BaggingClassifier(random_state=42),
                         {'n_estimators': [10,50,100], 'max_samples': [0.5,0.7,1.0], 'max_features': [0.5,0.7,1.0]}),
    'RandomForest':     (RandomForestClassifier(random_state=42),
                         {'n_estimators': [100,200], 'max_depth': [5,10,None], 'max_features': ['sqrt','log2']}),
    'AdaBoost':         (AdaBoostClassifier(random_state=42),
                         {'n_estimators': [50,100,200], 'learning_rate': [0.01,0.1,1.0]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42),
                         {'n_estimators': [100,200], 'learning_rate': [0.05,0.1,0.2], 'max_depth': [3,5]}),
    'XGBoost':          (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                         {'n_estimators': [100,200], 'learning_rate': [0.05,0.1,0.2], 'max_depth': [3,5,7]}),
}

for name, (model, params) in MODELS.items():
    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
        gs.fit(X_tr, y_tr)
        f1 = f1_score(y_te, gs.best_estimator_.predict(X_te))
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric('f1_score', f1)
        mlflow.sklearn.log_model(gs.best_estimator_, 'model', skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier'])
        print(f'{name} — F1: {f1:.4f}')
        if f1 > best_score:
            best_score, best_model_obj = f1, gs.best_estimator_

with open('/tmp/model.pkl',    'wb') as f: pickle.dump(best_model_obj, f)
with open('/tmp/features.pkl', 'wb') as f: pickle.dump(X_tr.columns.tolist(), f)
create_repo(repo_id=MODEL_REPO_ID, repo_type='model', exist_ok=True, token=HF_TOKEN)
for path, name in [('/tmp/model.pkl','model.pkl'),('/tmp/features.pkl','features.pkl')]:
    api.upload_file(path_or_fileobj=path, path_in_repo=name,
                    repo_id=MODEL_REPO_ID, repo_type='model', token=HF_TOKEN)
print(f'Best model (F1={best_score:.4f}) registered on Hugging Face Model Hub.')
