import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from CKA import CKA, CudaCKA
import random
from torch.nn import DataParallel
from utils.evaluate_llms_utils import generate_instruction_following_task_prompt, generate_code_task_prompt,\
                 read_mbpp, batch_data

#  获取每层输出
def get_hidden_states(model, tokenizer, text, device):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = [state.to(device) for state in outputs.hidden_states]  # 将每层的隐藏状态移动到同一设备上
    return hidden_states

def compute_cka(model1, model2, tokenizer1, tokenizer2, text, device):
        cuda_cka = CudaCKA(device)
        hidden_states1 = get_hidden_states(model1, tokenizer1, text, device)
        hidden_states2 = get_hidden_states(model2, tokenizer2, text, device)

        num_layers = len(hidden_states1)
        cka_layer_similarity = torch.zeros((num_layers, num_layers), device=device)

        for i in range(num_layers):
            for j in range(num_layers):
                with torch.no_grad():
                    feature1 = hidden_states1[i].squeeze().float()
                    feature2 = hidden_states2[j].squeeze().float()

                    # 计算CKA相似度
                    cka_similarity = cuda_cka.linear_CKA(feature1, feature2)
                    cka_layer_similarity[i][j] = cka_similarity

        return cka_layer_similarity

parser = argparse.ArgumentParser(description="Generate CKA for MBPP and AlpacaEval")
parser.add_argument("--model_path1", help="Path to the Model1 checkpoint")
parser.add_argument("--model_path2", help="Path to the Model2 checkpoint")
parser.add_argument("--cache_dir", default=None, help="Path to the Cache Dir")
parser.add_argument("--dataset_name", default="alpaca_eval", help="Dataset to use")
args = parser.parse_args()

model_path1 = args.model_path1
model_path2 = args.model_path2
cache_dir = args.cache_dir
dataset_name = args.dataset_name



tokenizer1 = AutoTokenizer.from_pretrained(model_path1, padding_side="left", cache_dir=cache_dir)
if not tokenizer1.pad_token:
    tokenizer1.pad_token = tokenizer1.unk_token
model1 = AutoModelForCausalLM.from_pretrained(model_path1, cache_dir=cache_dir, output_hidden_states=True)

tokenizer2 = AutoTokenizer.from_pretrained(model_path2, padding_side="left", cache_dir=cache_dir)
if not tokenizer2.pad_token:
    tokenizer2.pad_token = tokenizer2.unk_token
model2 = AutoModelForCausalLM.from_pretrained(model_path2, cache_dir=cache_dir, output_hidden_states=True)

model1.eval()
model2.eval()
device1 = torch.device('cuda:1')
device2 = torch.device('cuda:2')

# model1.to(device1)
# model2.to(device2)
device = torch.device('cuda:0')

if(dataset_name == "alpaca_eval"):
    # 2. 加载数据
    try:
        dataset = datasets.load_dataset(path=os.path.join(cache_dir, "alpaca_eval"), name="alpaca_eval")["eval"]
    except:
        dataset = datasets.load_dataset(path="tatsu-lab/alpaca_eval", name="alpaca_eval", cache_dir=cache_dir)["eval"]

    instructions = []
    for example in dataset:
        instructions.append(example["instruction"])
    text = instructions[random.randint(0, len(instructions))]
    
    cka_layer_similarity = compute_cka(model1, model2, tokenizer1, tokenizer2, generate_instruction_following_task_prompt(text), device)

    selected_matrix = cka_layer_similarity.cpu().numpy()


elif(dataset_name == "mbpp"):
    test_data_path = "./code_data/mbpp.test.jsonl"
    problems = read_mbpp(test_data_path)
    task_id = random.randint(0,len(problems))
    prompt = f"\n{problems[task_id]['text']}\nTest examples:"
    if task_id == 493:
        # The test examples are too long, we choose to only include the function name.
        test_example = problems[task_id]['test_list'][0]
        prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
    else:
        for test_example in problems[task_id]['test_list']:
            prompt += f"\n{test_example}"
    prompt = prompt.replace('    ', '\t')
    prompt = generate_code_task_prompt(prompt)
    selected_matrix =  compute_cka(model1, model2, tokenizer1, tokenizer2, prompt, device).cpu().numpy()



model1_name = model_path1.split("/")[-1]
model2_name = model_path2.split("/")[-1]
# 5. 使用Matplotlib和Seaborn绘制热力图
plt.figure(figsize=(12, 10))

sns.heatmap(selected_matrix, cbar=True,
            xticklabels=[f'Layer {i+1}' for i in range(selected_matrix.shape[1])],
            yticklabels=[f'Layer {i+1}' for i in range(selected_matrix.shape[0])])

plt.title('CKA Similarity Matrix between Two Models')
plt.xlabel(f'{model1_name} Layers')
plt.ylabel(f'{model2_name} Layers')
out_dir = f'cka_fig/between_models/{dataset_name}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
plt.savefig(f'cka_fig/between_models/{dataset_name}/{model1_name}_{model2_name}_cka_similarity_matrix.png')
