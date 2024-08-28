import torch
import yaml
import argparse
import uuid
import os
import json

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

parser = argparse.ArgumentParser(description="Perform Merging")
parser.add_argument("--drop_rate", type=float, help="Drop Rate of delta parameters")
parser.add_argument("--merge_method", help="merge-method")
parser.add_argument("--models", help="Models to Merge", choices = ["math_code", "LM_math", "LM_code", "LM_math_code", "LM", "math", "code", "Coder"])
parser.add_argument("--weights", default = 1, type=float, help="Merge Weights")
parser.add_argument("--lambda_factor", default = 1, type=float, help="Scaling Factor in Step 3")
parser.add_argument("--window_size", default = 0, type=float, help="Window Size for Probabilities. Set to 0 for TIES and DARE")
parser.add_argument("--rescale", default = 1, type=int, choices = [1,0], help="Whether to rescale in step 1")
parser.add_argument("--seed", default = 42, type=int, help="Random Seed")

args = parser.parse_args()


LORA_MERGE_CACHE = "/tmp"
COPY_TOKENIZER = True
LAZY_UNPICKLE = False
LOW_CPU_MEMORY = False

save_model_base = "/data/lsq"



models = ''
weight = args.weights
density = round(1 - args.drop_rate, 2)




# folder path to store the result in
if args.seed != 42:
  OUTPUT_PATH = f"{save_model_base}/save_merge_models/deepseek_{args.models}_{args.seed}_{args.merge_method}_{args.lambda_factor}_{args.drop_rate}_{args.window_size}_{args.weights}"
else:
  OUTPUT_PATH = f"{save_model_base}/save_merge_models/deepseek_{args.models}_{args.merge_method}_{args.lambda_factor}_{args.drop_rate}_{args.window_size}_{args.weights}"

if args.rescale == 0:
  OUTPUT_PATH += "_norescale"




if "LM" in args.models:
    models += f"""
  - model: {"deepseek-ai/deepseek-llm-7b-chat"}
    parameters:
      weight: {weight}"""

if "math" in args.models:
    models += f"""
  - model: {"deepseek-ai/deepseek-math-7b-base"}
    parameters:
      weight: {weight}"""

if "code" in args.models:
    models += f"""
  - model: {"deepseek-ai/deepseek-coder-7b-instruct-v1.5"}
    parameters:
      weight: {weight}"""


yaml_config = f"""
models:{models}
merge_method: {args.merge_method}
base_model: {"deepseek-ai/deepseek-llm-7b-base"}
dtype: float16
parameters:
  density: {density}
  lambda: {args.lambda_factor}
  window_size: {args.window_size}
  rescale: {args.rescale}
"""

if args.merge_method == "task_arithmetic":
  yaml_config = f"""
models:{models}
merge_method: {args.merge_method}
base_model: {"deepseek-ai/deepseek-llm-7b-base"}
dtype: float16
"""
  OUTPUT_PATH = f"{save_model_base}/deepseek_{args.models}_{args.merge_method}"
  
  
if args.merge_method == "dare_linear":
   yaml_config = f"""
models:{models}
merge_method: {args.merge_method}
base_model: {"deepseek-ai/deepseek-llm-7b-base"}
dtype: float16
parameters:
  density: {density}
  rescale: {args.rescale}
"""
   OUTPUT_PATH = f"{save_model_base}/deepseek_{args.models}_{args.merge_method}_{args.drop_rate}"

uniq_id = uuid.uuid4()
CONFIG_YML = f"./configs/{args.models}_{uniq_id}_{args.drop_rate}_{args.weights}.yml"
# Save config as yaml file
with open(CONFIG_YML, 'w', encoding="utf-8") as f:
    f.write(yaml_config)


print(f"Setting Seed to {args.seed}")
torch.manual_seed(args.seed)




with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

run_merge(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
    ),
)
print("Merge Done!")

print("Deleting Config File:")
os.remove(CONFIG_YML)

print("Done!")
