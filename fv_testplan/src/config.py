from types import SimpleNamespace
from pathlib import Path
from utils import get_user, get_host
from collections import OrderedDict
from datetime import datetime

# task = 'gen_plan'
task = 'build_KG'

if task == 'gen_plan':
    file_path = '/home/scratch.yunshengb_cpu/fv/fv_testplan/data/apb/apbi2c_spec.pdf'
    design_dir = '/home/scratch.yunshengb_cpu/assertion_data_for_LLM/verified_assertions/communication_controller_apb_to_i2c/apb/'

    # llm_model = "mistral" # ollama
    # llm_model = 'mixtral_8x7b'
    llm_model = 'gpt-35-turbo'
    # llm_model = 'gpt-4'
    # llm_model = 'gpt-4-turbo'

    llm_args = {}
    if llm_model == "mixtral_8x7b":
        llm_args = {
            'base_url': "https://chipnemo-nvcf-proxy-rc.sc-paas.nvidia.com/v1/internal/chipnemo/mixtral_8x7b_H100/",
            'api_key': "eyJhbGciOiJFUzUxMiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiI2NjNlYmE1NjllOWE3OGE2ZWQyZDQ0MTEiLCJleHAiOjE3MjM4NDM2ODEsInN1YiI6Imxpd2FuIiwidHlwIjoiVVNFUiIsInVzZXJJZCI6Imxpd2FuIiwib3JpZ2luIjoibGl3YW4iLCJqdGkiOiI2Njk4Mzc2MTY1MjZkM2I2ZjE0MzhjYjMiLCJpYXQiOjE3MjEyNTE2ODEsImlzcyI6IjYyOTYzMGRkMGJhN2Y3NDEzYTNiNjMwMSJ9.AZiLB-kczc5dzVDoIhVVPkKVfIpQayI6UlCWH3_YcBgCXirgSMWKFmXCxI7J9jt9ipK8LweGkUMa1QxLhLwa-IIgAeKd-yDz3hSezH776xvYscJNHWoSqed8E-zHx2t4UzUGNMwVyy8kTJni_c7k5cuVz2lKUUsFgN422CmcfrDXtL7Z",
        }
elif task == 'build_KG':


    # input_file_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/apbi2c_spec.pdf'
    input_file_path = f'/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/apb/apbi2c_spec_processed.jsonl'

    # input_file_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/DDI0413D_cortexm1_r1p0_trm/DDI0413D_cortexm1_r1p0_trm.pdf'
    # input_file_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/priv-isa-asciidoc_20240411/priv-isa-asciidoc_20240411.pdf
    # input_file_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/unpriv-isa-asciidoc_20240411/unpriv-isa-asciidoc_20240411.pdf'
    # input_file_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/hqa_validation/hqa_validation.txt'
    # input_file_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/hqa_validation_debug_2/hqa_validation_debug_2.txt'
    # input_file_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/data/hqa_validation/hqa_validation.txt'

    env_source_path = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/rag_apb/.env'

    settings_source_path = (
        '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/rag_apb/settings.yaml'
    )

    graphrag_local_dir = '/home/scratch.yunshengb_avr_misc/fv/fv_testplan/graphrag' # must use local version with our own modifications

user = get_user()
hostname = get_host()

###################################### Below: no need to touch ######################################

# Define the root path (adjust this if necessary)
ROOT = (
    Path(__file__).resolve().parents[2]
)  # Adjust this number based on actual .git location

try:
    import git
except Exception as e:
    raise type(e)(f'{e}\nRun pip install gitpython or\nconda install gitpython')
try:
    repo = git.Repo(ROOT)
    repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-1]
    local_branch_name = repo.active_branch.name
    commit_sha = repo.head.object.hexsha
except git.exc.InvalidGitRepositoryError as e:
    raise Exception(f"Invalid Git repository at {ROOT}") from e

proj_dir = ROOT

vars = OrderedDict(vars())
FLAGS = OrderedDict()
for k, v in vars.items():
    if not k.startswith('__') and type(v) in [
        int,
        float,
        str,
        list,
        dict,
        type(None),
        bool,
    ]:
        FLAGS[k] = v
FLAGS = SimpleNamespace(**FLAGS)
