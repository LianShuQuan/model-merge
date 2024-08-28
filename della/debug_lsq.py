import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import os
import sys
import time
import shutil
import logging
import glob
import datasets
import json
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems, stream_jsonl
import torch
import argparse
import jsonlines
from lm_eval.models.utils import stop_sequences_criteria
from utils.evaluate_llms_utils import generate_instruction_following_task_prompt, generate_code_task_prompt,\
                 read_mbpp, batch_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Results for MBPP and AlpacaEval")
    parser.add_argument("--model_path", help="Path to the Model checkpoint")
    parser.add_argument("--dataset", default = None, help="Evaluation Dataset to complete generations")

    args = parser.parse_args()

    model_name = args.model_path
    dataset_name = args.dataset


    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1")

    next(model.parameters())
    model_device = next(model.parameters()).device

    print(model.parameters())
    
    print("Model is on device:", model_device)