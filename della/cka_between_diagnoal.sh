#!/bin/bash

# ~Z~I模~^~K路~D对
model_pairs=(
  "layoric/llama-2-13b-code-alpaca WizardLMTeam/WizardLM-13B-V1.2"
  "layoric/llama-2-13b-code-alpaca vanillaOVO/WizardMath-13B-V1.0"
  "vanillaOVO/WizardMath-13B-V1.0 WizardLMTeam/WizardLM-13B-V1.2"
  "/home/lsq/della/LM_math_code_della_1.1_0.3_0.14_1.0 layoric/llama-2-13b-code-alpaca"
  "/home/lsq/della/LM_math_code_della_1.1_0.3_0.14_1.0 WizardLMTeam/WizardLM-13B-V1.2"
  "/home/lsq/della/LM_math_code_della_1.1_0.3_0.14_1.0 vanillaOVO/WizardMath-13B-V1.0"
  "deepseek-ai/deepseek-llm-7b-chat deepseek-ai/deepseek-math-7b-base"
  "deepseek-ai/deepseek-llm-7b-chat deepseek-ai/deepseek-coder-7b-instruct-v1.5"
  "deepseek-ai/deepseek-math-7b-base deepseek-ai/deepseek-coder-7b-instruct-v1.5"
  "/home/lsq/della/LM_math_code_dare_linear_1.1_0.3_0.0_1.0 deepseek-ai/deepseek-math-7b-base"
  "/home/lsq/della/LM_math_code_dare_linear_1.1_0.3_0.0_1.0 deepseek-ai/deepseek-llm-7b-chat"
  "/home/lsq/della/LM_math_code_dare_linear_1.1_0.3_0.0_1.0 deepseek-ai/deepseek-coder-7b-instruct-v1.5"
)

# 循~N~I~L~O个模~^~K对
for pair in "${model_pairs[@]}"; do
  set -- $pair  # ~F~W符串转~M为~M置~O~B~U
  model_path1=$1
  model_path2=$2

  # ~I~L~Q令
  python3 CKA_between_models.py --model_path1 $model_path1 --model_path2 $model_path2
done

for pair in "${model_pairs[@]}"; do
  set -- $pair  # ~F~W符串转~M为~M置~O~B~U
  model_path1=$1
  model_path2=$2

  # ~I~L~Q令
  python3 CKA_between_models.py --model_path1 $model_path1 --model_path2 $model_path2 --dataset_name mbpp
done