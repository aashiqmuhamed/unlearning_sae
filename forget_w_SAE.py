from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader
from sae_lens import SAE
from tqdm import tqdm

import os
from pathlib import Path
from utils import collate_fn,plot_importances,plot_results,get_shuffled_forget_retain_tokens,clean_importances
import pickle
from copy import deepcopy

class forget_w_SAE_CausalLM():
    def __init__(self, model_name,sae_name,sae_id,retain_dset,fgt_dset,use_error_term=True, device=None,num_activations=[100,50,25,10,5],th_ratio = 10**3,batch_size=1,input_model: AutoModelForCausalLM = None, input_tokenizer: AutoTokenizer= None):
        
        self.model_name = model_name
        self.sae_name = sae_name
        self.sae_id = sae_id
        self.device = device
        if self.device is None:
            print('Using cpu')
            self.device = "cpu"
        else:
            self.device = device
        
        if input_model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                                                          self.model_name,
                                                          device_map=self.device)
        else:
            self.model = input_model

        if input_tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = input_tokenizer

        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=self.sae_name,
            sae_id=self.sae_id, 
            device=self.device
        )

        self.sae.use_error_term = use_error_term
        self.hook_layer = self.cfg_dict["hook_layer"]
        print(self.hook_layer)
        
        #data
        self.fgt_dset, self.retain_dset = get_shuffled_forget_retain_tokens(tokenizer=self.tokenizer,forget_dataset=fgt_dset['text'],retain_dataset=retain_dset['text'])

        #params unlearning 
        self.num_activations = num_activations     
        self.th_ratio = th_ratio
        self.batch_size = batch_size

        self.model.register_buffer("flag_batch_activations", torch.tensor([False for _ in range(self.batch_size)], dtype=torch.bool))

    def hook_fn_activations(self, module, input, output):
        self.activations = output[0]

    def single_batch_importance(self,target_act):
        self.optimizer_FIM.zero_grad()
        torch.cuda.empty_cache()

        ####
        feat = self.sae.encode_standard(target_act)
        recon = self.sae.decode(feat)
        
        feat = feat>0
        if self.activations_sae is None:
            self.activations_sae = feat.sum(dim=(0,1))/feat.shape[1]
        else:
            self.activations_sae += feat.sum(dim=(0,1))/feat.shape[1]

        ####
        #simple
        #loss = self.criterion_FIM(recon[:,1:,:], target_act[:,1:,:])

        #select tokens which correspond to less than 5% of activations
        #this filter removes tokens which don't exihibit strong token/few-concept connection 
        feat_binary = (feat>0).sum(dim=2)/feat.shape[2]
        id = (feat_binary<0.05).squeeze()
        id[0] = False 
        loss = self.criterion_FIM(recon[:,id,:], target_act[:,id,:])
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
        print('DATALOADER len: ',self.N)
        self.importances = self.zerolike_params_dict(self.sae)
        hook = self.model.model.layers[self.hook_layer].register_forward_hook(self.hook_fn_activations)
        self.activations_sae = None
        for batch_tokens in tqdm(dataloader):

            with torch.no_grad():
                _ = self.model(batch_tokens)                
                target_act = self.activations


            self.single_batch_importance(target_act)


        self.optimizer_FIM.zero_grad()
        torch.cuda.empty_cache()
        hook.remove()
        self.activations_sae = self.activations_sae/float(self.N)
        return self.get_norm_importances()
    
    def get_activations_indexes(self,num_act_rem):
        num_act_rem_top = int(num_act_rem/3)
        num_act_rem_th = int(2*num_act_rem/3)
        key = 'W_dec'
        
        norm_vec_fgt = torch.norm(self.importances_fgt[key].data,dim=1)
        norm_vec_fgt[self.activations_sae_fgt<0.05] = 0

        norm_vec = torch.norm(self.importances_retain[key].data,dim=1)

        norm_vec_fgt_ratio = norm_vec_fgt/(norm_vec+10**-21)
        norm_vec_fgt_ratio[norm_vec==0] = 0

        id_ratio = (norm_vec_fgt_ratio<self.th_ratio)
        print(f'Feature available: {(id_ratio==False).sum()} over {id_ratio.shape[0]}')
        norm_vec_fgt_cp = deepcopy(norm_vec_fgt)
        norm_vec_fgt[id_ratio] = 0
        _,index_top = torch.sort(norm_vec_fgt)


        return index_top[-min(num_act_rem,(id_ratio==False).sum()):],norm_vec_fgt_cp,norm_vec

        
    def select_activations_FIM(self, optimizer = None, criterion = torch.nn.MSELoss()):
        
        dataloader_fgt = DataLoader(self.fgt_dset, batch_size=self.batch_size, shuffle=True,)
        dataloader_retain = DataLoader(self.retain_dset, batch_size=self.batch_size, shuffle=True,)
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
                self.activations_sae_fgt = imp_dict['activations'].to(self.device)
            
            for key in self.importances_fgt.keys():
                self.importances_fgt[key] = self.importances_fgt[key].to(self.device)
            print("Importances fgt loaded from cache")
        else:
            self.importances_fgt = self.compute_FIM(dataloader_fgt)
            #save
            with open(cache_file_fgt, 'wb') as f:
                pickle.dump({'importances': clean_importances(deepcopy(self.importances_fgt)),
                             'activations': self.activations_sae.detach().cpu()
                             }, f)
            print("Importances fgt saved to cache")
            self.activations_sae_fgt = self.activations_sae
            self.activations_sae =  None

        if os.path.exists(cache_file_rtn):
            with open(cache_file_rtn, 'rb') as f:
                imp_dict = pickle.load(f)
                self.importances_retain = imp_dict['importances']
                self.activations_sae_retain = imp_dict['activations'].to(self.device)

            for key in self.importances_retain.keys():
                self.importances_retain[key] = self.importances_retain[key].to(self.device)
            print("Importances retain loaded from cache")
        else:
            self.importances_retain = self.compute_FIM(dataloader_retain)
            #save
            with open(cache_file_rtn, 'wb') as f:
                pickle.dump({'importances': clean_importances(deepcopy(self.importances_retain)),
                             'activations': self.activations_sae.detach().cpu()
                             }, f)
            print("Importances rtn saved to cache")
            self.activations_sae_retain = self.activations_sae
            self.activations_sae =  None

        

        #clean stuff
        del self.criterion_FIM,self.optimizer_FIM
        self.dict_indexes = {}
        for num_act_rem in self.num_activations:
            self.dict_indexes[f'N_{num_act_rem}'],norm_fgt,norm_rtn = self.get_activations_indexes(num_act_rem)

            print('plot importances')
            plot_importances(norm_fgt.detach().cpu().numpy(),
                             norm_rtn.detach().cpu().numpy(),
                             indexes = self.dict_indexes[f'N_{num_act_rem}'].detach().cpu().numpy(),
                             filename=os.path.join('importances', f"{self.model_name.replace('/','_')}_{self.sae_id.replace('/','_')}_N_{num_act_rem}_th_{self.th_ratio}.png"))
        

    
    def compute_activation_freq(self,dataloader,num_act_rem=50,flag_tokenizer=True):
        self.N = len(dataloader)
        print('DATALOADER len: ',self.N)
        overall_cnt = []
        with torch.no_grad():
            hook = self.model.model.layers[self.hook_layer].register_forward_hook(self.hook_fn_activations)
        
            for batch in tqdm(dataloader):
                if flag_tokenizer:
                    batch_tokens = self.tokenizer(batch, return_tensors="pt",truncation=True,max_length=1024,add_special_tokens=False).input_ids.to(self.device) 
                else:
                    batch_tokens = batch             
                _ = self.model(batch_tokens)
                target_act = self.sae.encode(self.activations)[:,1:,self.dict_indexes[f'N_{num_act_rem}']]
                target_act= (target_act>0)
                val = target_act.sum()/(target_act.shape[1]*target_act.shape[2])

                overall_cnt.append(val.item())

        hook.remove()

        return overall_cnt
    
    #add error term
    def add_sae_hook(self,mod,inputs,outputs): 
        activation = self.sae.encode(outputs[0])
        feature_acts_clean = deepcopy(activation)
        #import pdb; pdb.set_trace()
        
        if activation.shape[1] ==1:
            st = 0
        else:
            st = 1
   
        buff = activation[:,st:,self.index_to_rem]

        # import pdb; pdb.set_trace()
        # if st==1 and (buff>0).sum()/(buff.shape[1]*buff.shape[2]) > 0.05:
        #     buff[buff>0]= self.clamp_val
        #     self.model.flag_batch_activation.fill_(True)
        # elif st==1 and (buff>0).sum()/(buff.shape[1]*buff.shape[2]) < 0.05:
        #     self.model.flag_batch_activation.fill_(False)

 
        # if self.model.flag_batch_activation:    
        #     buff[buff>0]= self.clamp_val
        ##########################################################
        #dynamic selection of activations
        ##########################################################
        if st==1:
            self.model.flag_batch_activation = ((buff>0).sum(dim=(1,2))/(buff.shape[1]*buff.shape[2])>0.01)
            print((buff>0).sum(dim=(1,2))/(buff.shape[1]*buff.shape[2]))
        index_active = (buff>0)
        index_active[self.model.flag_batch_activation==False,:,:] = False
        buff[index_active] = self.clamp_val
        ##########################################################
        activation[:,st:,self.index_to_rem] = buff
        sae_out = self.sae.decode(activation)
        with torch.no_grad():

            reconstruct_clean = self.sae.decode(feature_acts_clean)
            sae_error =(outputs[0] - reconstruct_clean)
            sae_error[self.model.flag_batch_activation==True,:,:] = 0
        sae_out = sae_out + sae_error
        
        return (sae_out,outputs[1])

    def get_model_with_sae(self,config_num,clamp_val):
        self.clamp_val = clamp_val
        self.select_activations_FIM()
        self.index_to_rem = self.dict_indexes[f'N_{self.num_activations[config_num]}']

        hook = self.model.model.layers[self.hook_layer].register_forward_hook(self.add_sae_hook)

        return hook 

if __name__ == "__main__":
    from datasets import load_dataset,concatenate_datasets
    from lm_eval import evaluator
    from lm_eval.utils import make_table
    from lm_eval.tasks import TaskManager

    from lm_eval.models.huggingface import HFLM
    import pandas as pd

    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    

    corpora_fgt = load_dataset('json',data_files="/home/jb/Documents/unlearning_sae/data_wmdp_forget_corpora/bio_remove_dataset.jsonl")['train']
    corpora_fgt.shuffle(seed=42)

    wmdp_bio = load_dataset("cais/wmdp", name="wmdp-bio", split="test")

    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    #filter wikitext for sentences <50
    wikitext = wikitext.filter(lambda x: len(x["text"])>50)
    wikitext.shuffle(seed=42)
 

    mmlu_history = load_dataset("cais/mmlu", "high_school_us_history", split="test")
    mmlu_history = concatenate_datasets([load_dataset("cais/mmlu", "philosophy", split="test"), load_dataset("cais/mmlu", "high_school_european_history", split="test"), mmlu_history])

    constructor = forget_w_SAE_CausalLM(model_name = "google/gemma-2-2b-it",
                                        sae_name = "gemma-scope-2b-pt-res-canonical",
                                        sae_id="layer_7/width_65k/canonical",
                                        retain_dset=wikitext,
                                        fgt_dset=corpora_fgt.take(2000),
                                        use_error_term=True,
                                        device="cuda:0",
                                        th_ratio=10**2)

    all_results = []
    for config_num_act in [1]:#range(len(constructor.num_activations)):
        for clamp_val in [-10,-50,-100,-200]:#,
            print(f'CONFIG ----> # of feat rem. {constructor.num_activations[config_num_act]}, clamp val {clamp_val}')
            hook_added = constructor.get_model_with_sae(config_num_act,clamp_val)
            lm_model = HFLM(pretrained=constructor.model,tokenizer=constructor.tokenizer)
            task_manager = TaskManager('INFO',include_path=None)
            results = evaluator.simple_evaluate(model= lm_model, tasks= ['wmdp_bio'], num_fewshot= None, batch_size= 2)
            
            print(make_table(results))
            acc_wmdp = results['results']['wmdp_bio']['acc,none']
            std_wmdp = results['results']['wmdp_bio']['acc_stderr,none']

            results = evaluator.simple_evaluate(model= lm_model, tasks= ['mmlu_jb'], num_fewshot= None, batch_size= 2)
            print(make_table(results))
            #remove hook 
            hook_added.remove()
            
            acc_mmlu = results['results']['mmlu_jb']['acc,none']
            std_mmlu = results['results']['mmlu_jb']['acc_stderr,none']

            all_results.append([constructor.th_ratio,clamp_val,constructor.num_activations[config_num_act],acc_wmdp,std_wmdp,acc_mmlu,std_mmlu])
    
    df = pd.DataFrame(all_results,columns=['th_ratio','clamp_val','num_activations','acc_wmdp','std_wmdp','acc_mmlu','std_mmlu'])
    #save df to csv
    Path('results_forget_w_SAE').mkdir(parents=True, exist_ok=True)
    file_to_save = os.path.join('results_forget_w_SAE',f'results_forget_w_SAE_th_ratio_{constructor.th_ratio}.csv')
    df.to_csv(file_to_save,index=False)
    plot_results()