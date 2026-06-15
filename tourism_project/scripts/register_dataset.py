import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN        = os.environ['HF_TOKEN']
HF_USERNAME     = os.environ['HF_USERNAME']
DATASET_REPO_ID = f'{HF_USERNAME}/visit-with-us-tourism'
api = HfApi()

create_repo(repo_id=DATASET_REPO_ID, repo_type='dataset', exist_ok=True, token=HF_TOKEN)
api.upload_file(
    path_or_fileobj='tourism_project/data/tourism.csv',
    path_in_repo='data/tourism.csv',
    repo_id=DATASET_REPO_ID,
    repo_type='dataset',
    token=HF_TOKEN
)
print(f'Dataset registered: https://huggingface.co/datasets/{DATASET_REPO_ID}')
