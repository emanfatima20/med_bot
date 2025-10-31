from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

# ---- 1️⃣ Local model path ----
model_path = r"C:\Users\PMYLS\Downloads\med_bot\smollm2_pubmed_full_v3"

# Make sure this folder exists and contains:
# config.json
# pytorch_model.bin (or *.bin)
# tokenizer.json or vocab files

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Local model path does not exist: {model_path}")

# ---- 2️⃣ Load tokenizer from Hugging Face online ----
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    use_fast=True
)

# ---- 3️⃣ Load model using AutoConfig to avoid repo validation ----
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, local_files_only=True)

# ---- 4️⃣ Create pipeline ----
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=300,
    temperature=0.7,
    do_sample=True,
    device=0 if torch.cuda.is_available() else -1
)

# ---- 5️⃣ Example usage ----
question = "What is the insulin?"
instruction = (
    f"### Instruction:\n"
    f"You are a medical assistant. Provide a short and clear answer.\n"
    f"Answer the following medical question:\n{question}\n\n### Response:"
)

result = pipe(
    instruction,
    max_new_tokens=200,     # force output beyond prompt
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)[0]["generated_text"]

print(result)
