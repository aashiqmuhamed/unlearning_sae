import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import glob

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

def pad_sequences(sequences, padding_value=0):
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    return padded_sequences

def collate_fn(batch):
    texts = []
    for item in batch:
        #import pdb; pdb.set_trace()
        if 'choices' in item:
            task_description = "The following are multiple choice questions (with answers).\n\n"            
            prompts = []#
            prompt = task_description + f"{item['question']}\nA: {item['choices'][0]}\nB: {item['choices'][1]}\nC: {item['choices'][2]}\nD: {item['choices'][3]}\nAnswer:"
            texts.append(prompt)
        elif 'text' in item:
            texts.append(item['text'])
        else:
            texts.append(item)
    # tokenized = tokenizer(texts, add_special_tokens=True, padding=False, truncation=True)
    # tokenized['input_ids'] = torch.tensor(pad_sequences(tokenized['input_ids']))
    # tokenized['attention_mask'] = torch.tensor(pad_sequences(tokenized['attention_mask']))
    return texts


def plot_importances(importance_dict,importance_dict_fgt):
    
    fig,ax = plt.subplots(figsize=(10, 6),ncols=2)

    for mat_name in [('W_enc',0),('W_dec',1)]:

        arr = importance_dict[mat_name[0]].detach().cpu().numpy().flatten()

        arr_feat = np.max(importance_dict[mat_name[0]].detach().cpu().numpy(),axis=mat_name[1])#.mean(axis=mat_name[1])

        arr_fgt = importance_dict_fgt[mat_name[0]].detach().cpu().numpy().flatten()

        arr_feat_fgt = np.max(importance_dict_fgt[mat_name[0]].detach().cpu().numpy(),axis=mat_name[1])#.mean(axis=mat_name[1])

        color = np.zeros_like(arr_feat_fgt)
        arr_feat_fgt_cp = arr_feat_fgt.copy()
        #arr_feat_fgt_cp[arr_feat>10**-9] = 0

        arr_feat_fgt_cp = (arr_feat_fgt_cp/(arr_feat+10**-21))
        arr_feat_fgt_cp[arr_feat==0] = 0
        index = np.argsort(arr_feat_fgt_cp)#(arr_feat_fgt_cp>10**3)#
        
        id_ratio = (arr_feat_fgt_cp<10**3)
        arr_feat_fgt_cp2 = arr_feat_fgt.copy()
        arr_feat_fgt_cp2[id_ratio] = 0
        id_val = np.argsort(arr_feat_fgt_cp2)

        color[index[-500:]] = 1
        color[id_val[-200:]] = 2
        #color[index[-400:-100]] = 3      
        # color[(arr_feat_fgt/(arr_feat+0.0000000001))>10] = 1
        # color[(arr_feat_fgt/(arr_feat+0.0000000001))>100] = 2
        # color[(arr_feat_fgt/(arr_feat+0.0000000001))>1000] = 3
        c_list = ['blue', 'red','green','black']
        c_plot = [c_list[int(color[i])] for i in range(len(color))]
        ax[mat_name[1]].scatter(arr_feat, arr_feat_fgt,s=1,c=c_plot)
        ax[mat_name[1]].set_title(mat_name[0])

    for e in [ax[0],ax[1]]:
        min_val = min(e.get_xlim()[0], e.get_ylim()[0])
        max_val = max(e.get_xlim()[1], e.get_ylim()[1])
    # Plot the diagonal line
        e.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Diagonal')
        e.set_ylabel('Fgt.')
        e.set_xlabel('Ret.')
        e.set_yscale('log')
        e.set_xscale('log')
    plt.savefig(f'ret_vs_fgt_mat_col.png')

    # fig,ax = plt.subplots(figsize=(10, 6),ncols=2)

    # for mat_name in [('W_enc',0),('W_dec',1)]:
    #     arr = importance_dict[mat_name[0]].detach().cpu().numpy().flatten()

    #     arr_feat = importance_dict[mat_name[0]].detach().cpu().numpy().sum(axis=0)

    #     arr_fgt = importance_dict_fgt[mat_name[0]].detach().cpu().numpy().flatten()

    #     arr_feat_fgt = importance_dict_fgt[mat_name[0]].detach().cpu().numpy().sum(axis=0)

    #     color = np.zeros_like(arr)
    #     color[(arr_fgt/(arr+0.0000000001))>1000] = 1
    #     c_list = ['blue', 'red']
    #     c_plot = [c_list[int(color[i])] for i in range(len(color))]

    #     ax[mat_name[1]].scatter(arr, arr_fgt,s=1,c=c_plot)
    #     ax[mat_name[1]].set_title(mat_name[0])

    # for e in [ax[0],ax[1]]:
    #     min_val = min(e.get_xlim()[0], e.get_ylim()[0])
    #     max_val = max(e.get_xlim()[1], e.get_ylim()[1])
    # # Plot the diagonal line
    #     e.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Diagonal')
    #     e.set_ylabel('Fgt.')
    #     e.set_xlabel('Ret.')
    #     e.set_yscale('log')
    #     e.set_xscale('log')
    # plt.savefig(f'ret_vs_fgt_mat.png')



def plot_results():
    #collect df
    files = glob.glob('results_forget_w_SAE/*.csv')
    print(files)
    df = None
    for file in files:
        if df is None:
            df = pd.read_csv(file)
        else:
            df = pd.concat([df, pd.read_csv(file)], ignore_index=True)
    print(df.head())
    # filter for th and #of feat
    th_list = df.th_ratio.unique()
    num_feat_list = df.num_activations.unique()
    fig,ax = plt.subplots(figsize=(10, 6),ncols=1,nrows=1)
    for th in th_list:
        for num_feat in num_feat_list:
            df_th = df[df.th_ratio == th]
            df_th = df_th[df_th.num_activations == num_feat]

            #plot
            
            ax.plot(df_th['acc_wmdp'],df_th['acc_mmlu'],label=f'th={th},#_feat={num_feat}',marker='o')
            # Add values at the top of each plot point
            print(df_th)
            # for i, value in enumerate(df_th['clamp_val'].values):
            #    plt.text(df_th['acc_wmdp'].values[i],df_th['acc_mmlu'].values[i]-df_th['acc_mmlu'].values[i]*0.02, str(value), ha='center', va='bottom')
    #add legend
    ax.legend()
    #save
    plt.savefig(f'acc_wmdp_vs_acc_mmlu.png')

if __name__ == '__main__':
    plot_results()  