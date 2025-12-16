from fastapi import FastAPI, Body
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# --- Configuration Constants ---
BASE_MODEL_NAME = "gpt2"
# Set device explicitly to 'cpu' for now, matching your VM setup
DEVICE = "cpu" 
LORA_DIR = "./lora-devotee"

# -------------------------
# App Setup
# -------------------------
app = FastAPI(title="Devotee Chat API")

# -------------------------
# Load tokenizer & base model (once on startup)
# -------------------------
print("Loading tokenizer...")
# Gemma requires trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model: {BASE_MODEL_NAME} (this may take time)...")

# Use float32 for CPU compatibility, set device_map=None
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32, 
    device_map=None,
    trust_remote_code=True
)

# Explicitly move model to CPU after loading
model.to(DEVICE)
model.eval() # Set model to evaluation mode

print(f"Model loaded successfully on {DEVICE}")


# -------------------------
# Request schema
# -------------------------
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7

# -------------------------
# Health endpoint
# -------------------------
@app.get("/")
def health():
    return {"message": "Devotee Chat API running!"}

# -------------------------
# Chat endpoint
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    prompt = req.prompt.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# -------------------------
# Load adapter endpoint (from the challenge requirements)
# -------------------------
@app.post("/load_adapter")
def load_adapter(adapter_path: str = Body(..., embed=True)):
    global model

    try:
        print(f"Loading adapter from: {adapter_path}")
        # Note: This assumes LORA_DIR contains your specific trained adapters
        model = PeftModel.from_pretrained(model, LORA_DIR) 
        model.to(DEVICE)
        model.eval()

        return {
            "status": "ok",
            "message": f"Adapter loaded from {LORA_DIR}"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


