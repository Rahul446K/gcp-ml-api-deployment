from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# ---------- FastAPI setup ----------
app = FastAPI(title="Devotee Chat API")

# ---------- Request body schema ----------
class ChatRequest(BaseModel):
    prompt: str

# ---------- Load base model + LoRA on startup ----------
BASE_MODEL_NAME = "tiiuae/falcon-7b-instruct"
LORA_DIR = "./lora-devotee"  # path inside the Docker image

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (this can take a while)...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,   # Optimized for CPU performance/compatibility
    device_map=None              # Set to None to avoid 'meta' device error
)
base_model = base_model.to('cpu') # Explicitly move the entire model to CPU

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR,
    device_map=None
)
model = model.to('cpu') # Ensure the final model is on CPU

model.config.pad_token_id = tokenizer.pad_token_id

device = next(model.parameters()).device
print("Model is on device:", device) # This should print 'cpu'

# ---------- Simple health endpoint ----------
@app.get("/")
def home():
    return {"message": "Devotee Chat API running!"}

# ---------- Chat endpoint ----------
@app.post("/chat")
def chat(req: ChatRequest):
    user_prompt = req.prompt

    # Format similar to training style
    full_prompt = f"User: {user_prompt}\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # --- FIX: Decode the first (and only) sequence in the batch ---
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # -------------------------------------------------------------
    return {"response": text}
