import os
from transformers import AutoTokenizer, AutoModel

# 1. Define model name and local path
model_id = "sentence-transformers/all-MiniLM-L6-v2"
local_dir = "./local_minilm"

# 2. Download from Hugging Face
print("Downloading model from Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# 3. Save directly to your current directory
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model successfully saved locally in: {os.path.abspath(local_dir)}")
