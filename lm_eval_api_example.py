import sys
from lm_eval import evaluator
import torch
import argparse
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager

from lm_eval.models.huggingface_custom import HFLM_custom as HFLM
from sae_lens import  HookedSAETransformer

input_model = HookedSAETransformer.from_pretrained(
            'google/gemma-2-2b-it',
            device='cuda:0',
            dtype=torch.bfloat16,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )

model = HFLM(pretrained='google/gemma-2-2b-it',hooked_model=input_model)

task_manager = TaskManager('INFO',include_path=None)
results = evaluator.simple_evaluate(
    model= model, tasks= ['wmdp_bio'], num_fewshot= None, batch_size= 2)
print(make_table(results))

