from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader
from sae_lens import SAE
from tqdm import tqdm

import os
from pathlib import Path
from utils import collate_fn,plot_importances
import pickle


class forget_w_SAE_CausalLM():
    def __init__(self, model_name,sae_name,sae_id,retain_dset,fgt_dset,use_error_term=True, device=None,num_activations=[750]):
        
        self.model_name = model_name
        self.sae_name = sae_name
        self.sae_id = sae_id
        self.device = device
        if self.device is None:
            print('Using cpu')
            self.device = "cpu"
        else:
            self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
                                                          self.model_name,
                                                          device_map=self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=self.sae_name,
            sae_id=self.sae_id, 
            device=self.device
        )
        self.sae.use_error_term = use_error_term
        self.hook_layer = self.cfg_dict["hook_layer"]
        
        #data
        self.retain_dset = retain_dset
        self.fgt_dset = fgt_dset
        #params unlearning 
        self.num_activations = num_activations     
    
    def hook_fn_activations(self, module, input, output):
        self.activations = output[0]

    def single_batch_importance(self,target_act):
        self.optimizer_FIM.zero_grad()
        torch.cuda.empty_cache()
        #JB check between normal forward which contains error term 
        ####
        feat = self.sae.encode(target_act)
        recon = self.sae.decode(feat)
        ####
        loss = self.criterion_FIM(recon[:,4:,:], target_act[:,4:,:])
        loss.backward()

        for (k1, p), (k2, imp) in zip(self.sae.named_parameters(), self.importances.items()):
            if p.grad is not None:
                imp.data += p.grad.data.clone().pow(2)



    def get_norm_importances(self):
        for _, imp in self.importances.items():
            imp.data /= float(self.N)
        return self.importances


    def zerolike_params_dict(self, model):
        """
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        """
        return dict(
                    [(k, torch.zeros_like(p, device=p.device)) for k, p in model.named_parameters()]
                   )

    def compute_FIM(self,dataloader):
        self.N = len(dataloader)
        self.importances = self.zerolike_params_dict(self.sae)
        hook = self.model.model.layers[self.hook_layer].register_forward_hook(self.hook_fn_activations)
        
        for batch in tqdm(dataloader):
            #max_lenght set as rmu
            batch_tokens = self.tokenizer(batch, return_tensors="pt",truncation=True, max_length=768).input_ids.to(self.device)
            with torch.no_grad():
                _ = self.model(batch_tokens)#, prepend_bos=True)
                #import pdb; pdb.set_trace()
                target_act = self.activations
                #check self.activations

            self.single_batch_importance(target_act)


        self.optimizer_FIM.zero_grad()
        torch.cuda.empty_cache()
        
        hook.remove()
        return self.get_norm_importances()
    
    def get_activations_indexes(self,num_act_rem):
        num_act_rem_top = int(num_act_rem/3)
        num_act_rem_th = int(2*num_act_rem/3)
        key = 'W_dec'
        norm_vec_fgt = torch.norm(self.importances_fgt[key].data,dim=1)
        norm_vec = torch.norm(self.importances_retain[key].data,dim=1)
        #print('NUM el', (norm_vec==0).sum())
        norm_vec_fgt_ratio = norm_vec_fgt/(norm_vec+10**-21)
        norm_vec_fgt_ratio[norm_vec==0] = 0
        _,index = torch.sort(norm_vec_fgt_ratio)

        id_ratio = (norm_vec_fgt_ratio<10**3)
        norm_vec_fgt[id_ratio] = 0
        _,index_top = torch.sort(norm_vec_fgt)
        index_clean = index[~torch.isin(index,index_top[-num_act_rem_top:])]

        return torch.cat((index_clean[-num_act_rem_th:],index_top[-num_act_rem_top:]),dim=0)


        
    def select_activations_FIM(self, optimizer = None,batch_size=1, criterion = torch.nn.MSELoss()):
        batch_size=batch_size
        dataloader_fgt = DataLoader(self.fgt_dset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        dataloader_retain = DataLoader(self.retain_dset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        if optimizer is None:
            self.optimizer_FIM = torch.optim.SGD(self.sae.parameters(), lr=0.1)
        self.criterion_FIM = criterion

        Path('importances').mkdir(parents=True, exist_ok=True)
        cache_file_fgt = os.path.join('importances', f"{self.model_name.replace('/','_')}_{self.sae_id.replace('/','_')}_imp_dict_fgt.pkl")
        cache_file_rtn = os.path.join('importances', f"{self.model_name.replace('/','_')}_{self.sae_id.replace('/','_')}_imp_dict_rtn.pkl")

        if os.path.exists(cache_file_fgt):
            with open(cache_file_fgt, 'rb') as f:
                imp_dict = pickle.load(f)
                self.importances_fgt = imp_dict['importances']
            print("Importances fgt loaded from cache")
        else:
            self.importances_fgt = self.compute_FIM(dataloader_fgt)
            #save
            with open(cache_file_fgt, 'wb') as f:
                pickle.dump({'importances': self.importances_fgt}, f)
            print("Importances fgt saved to cache")

        if os.path.exists(cache_file_rtn):
            with open(cache_file_rtn, 'rb') as f:
                imp_dict = pickle.load(f)
                self.importances_retain = imp_dict['importances']
            print("Importances retain loaded from cache")
        else:
            self.importances_retain = self.compute_FIM(dataloader_retain)
            #save
            with open(cache_file_rtn, 'wb') as f:
                pickle.dump({'importances': self.importances_retain}, f)
            print("Importances rtn saved to cache")

        plot_importances(self.importances_retain,self.importances_fgt)

        #clean stuff
        del self.criterion_FIM,self.optimizer_FIM
        self.dict_indexes = {}
        for num_act_rem in self.num_activations:
            self.dict_indexes[f'N_{num_act_rem}'] = self.get_activations_indexes(num_act_rem)

    def add_sae_hook(self,mod,inputs,outputs): 
        activation = self.sae.encode(outputs[0])
        buff = activation[:,:,self.index_to_rem]
        buff[buff>0]= -100#preliminar
        activation[:,:,self.index_to_rem] = buff
        return (self.sae.decode(activation),outputs[1])

    def get_model_with_sae(self):
        self.select_activations_FIM()
        self.index_to_rem = self.dict_indexes[f'N_{self.num_activations[0]}']

        hook = self.model.model.layers[self.hook_layer].register_forward_hook(self.add_sae_hook)

    #def unlearn_by_FIM(self,)

if __name__ == "__main__":
    from datasets import load_dataset
    from lm_eval import evaluator
    from lm_eval.utils import make_table
    from lm_eval.tasks import TaskManager

    from lm_eval.models.huggingface import HFLM

    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    wikitext.shuffle(seed=42)

    corpora_fgt = load_dataset('json',data_files="/home/jb/Documents/unlearning_sae/data_wmdp_forget_corpora/bio_remove_dataset.jsonl")['train']
    corpora_fgt.shuffle(seed=42)
    constructor = forget_w_SAE_CausalLM(model_name = "google/gemma-2-2b-it",
                                        sae_name = "gemma-scope-2b-pt-res-canonical",
                                        sae_id="layer_7/width_65k/canonical",
                                        retain_dset=wikitext.take(1000),
                                        fgt_dset=corpora_fgt.take(1000),
                                        use_error_term=True,
                                        device="cuda:0")


    
    constructor.get_model_with_sae()
    lm_model = HFLM(pretrained=constructor.model,tokenizer=constructor.tokenizer)
    task_manager = TaskManager('INFO',include_path=None)
    results = evaluator.simple_evaluate(model= lm_model, tasks= ['wmdp_bio'], num_fewshot= None, batch_size= 2)
    print(make_table(results))
    results = evaluator.simple_evaluate(model= lm_model, tasks= ['mmlu_jb'], num_fewshot= None, batch_size= 2)
    print(make_table(results))