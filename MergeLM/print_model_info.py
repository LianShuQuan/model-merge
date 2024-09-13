from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

model_path1 = "vanillaOVO/WizardMath-13B-V1.0"
model1:nn.Module = AutoModelForCausalLM.from_pretrained(model_path1)

tokenizer1 = AutoTokenizer.from_pretrained(model_path1)

for name, param in model1.named_parameters():
    print(name)
