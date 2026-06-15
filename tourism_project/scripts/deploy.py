import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN      = os.environ['HF_TOKEN']
HF_USERNAME   = os.environ['HF_USERNAME']
SPACE_REPO_ID = f'{HF_USERNAME}/visit-with-us-wellness-app'
api = HfApi()

create_repo(repo_id=SPACE_REPO_ID, repo_type='space', space_sdk='docker', exist_ok=True, token=HF_TOKEN)
for local_f, repo_f in [
    ('tourism_project/deployment/app.py',           'app.py'),
    ('tourism_project/deployment/requirements.txt', 'requirements.txt'),
    ('tourism_project/deployment/Dockerfile',       'Dockerfile')
]:
    api.upload_file(path_or_fileobj=local_f, path_in_repo=repo_f,
                    repo_id=SPACE_REPO_ID, repo_type='space', token=HF_TOKEN)
    print(f'Uploaded {repo_f}')
print(f'App live at: https://huggingface.co/spaces/{SPACE_REPO_ID}')
