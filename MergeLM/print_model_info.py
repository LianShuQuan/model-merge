from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

model_path1 = "deepseek-ai/deepseek-llm-7b-base"
model1:nn.Module = AutoModelForCausalLM.from_pretrained(model_path1)

tokenizer1 = AutoTokenizer.from_pretrained(model_path1)

for name, param in model1.named_parameters():
    print(name)
