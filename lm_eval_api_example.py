import sys
from lm_eval import evaluator
import torch
import argparse
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager

from lm_eval.models.huggingface_custom import HFLM_custom as HFLM

model = HFLM(pretrained='google/gemma-2-2b-it')

task_manager = TaskManager('INFO',include_path=None)
results = evaluator.simple_evaluate(
    model= model, tasks= ['wmdp'], num_fewshot= None, batch_size= 2)
print(make_table(results))

#max_batch_size= None, device =None, use_cache= None, limit= None ,check_integrity =False, write_out= False, log_samples= False,  system_instruction= None, apply_chat_template= False, fewshot_as_multiturn = False, gen_kwargs= None, task_manager=task_manager, verbosity= 'INFO', predict_only =False, random_seed= 0, numpy_random_seed =1234, torch_random_seed= 1234, fewshot_random_seed= 1234, cache_requests= False, rewrite_requests_cache=False, delete_requests_cache= False