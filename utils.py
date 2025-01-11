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

def plot_importances(norm_fgt,norm_rtn,indexes,filename):
    color = np.zeros_like(norm_fgt)
    color[indexes] = 1
    c_list = ['blue', 'red','green','black']
    c_plot = [c_list[int(color[i])] for i in range(len(color))]

    fig,ax = plt.subplots(figsize=(10, 6),ncols=1,nrows=1)
    ax.scatter(norm_rtn,norm_fgt,s=1,c=c_plot)

    for e in [ax]:
        min_val = min(e.get_xlim()[0], e.get_ylim()[0])
        max_val = max(e.get_xlim()[1], e.get_ylim()[1])
    # Plot the diagonal line
        e.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Diagonal')
        e.set_ylabel('Fgt.')
        e.set_xlabel('Ret.')
        e.set_yscale('log')
        e.set_xscale('log')
    plt.savefig(filename)
    plt.close()

def plot_results(input_folder):
    #collect df
    files = glob.glob(input_folder+'/*.csv')
    print('files csv collectd ',files)
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
            # for i, value in enumerate(df_th['clamp_val'].values):
            #    plt.text(df_th['acc_wmdp'].values[i],df_th['acc_mmlu'].values[i]-df_th['acc_mmlu'].values[i]*0.02, str(value), ha='center', va='bottom')
    #add legend
    ax.legend()
    plt.ylim(0.6,1)
    plt.xlim(0,1)
    #save
    print('plot_saved in: ',f'{input_folder}acc_wmdp_vs_acc_mmlu.png')
    plt.savefig(f'{input_folder}/acc_wmdp_vs_acc_mmlu.png')
    plt.close()
if __name__ == '__main__':
    plot_results(input_folder='/home/jb/Documents/unlearning_sae/results_FIM_SAE/SAEBench/')  