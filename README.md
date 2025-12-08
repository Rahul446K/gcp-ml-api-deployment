# gcp-ml-api-deployment
Deployment project of fine-tune a 7B parameter LLM AI model, exposed as a production-ready API endpoint on GCP using Docker and FastAPI.


# GCP ML API Deployment – Falcon-7B LoRA Adapter  
Deploying a fine-tuned 7B LLM as a production-ready REST API using Google Cloud, Docker, and FastAPI.

This project demonstrates the full lifecycle of taking a large-scale language model (Falcon-7B-Instruct), fine-tuning it using LoRA, and deploying it as a callable API endpoint on a CPU-only Google Cloud VM.  
It includes training, optimization, containerization, networking, and API exposure.

---

## 📌 Project Repository  
GitHub: https://github.com/Rahul446K/gcp-ml-api-deployment.git

---

# 🚀 Overview

The primary goal of this project is to fine-tune a 7B parameter model and run it in a **resource-constrained CPU environment**.  
Since 7B models normally require a GPU, several optimizations were introduced:

- Training LoRA adapters on **Kaggle GPU** due to GCP GPU unavailability  
- Running inference on **CPU** using optimized memory loading  
- Utilizing **FastAPI** to expose model inference as REST endpoints  
- Packaging the entire service in **Docker** for reproducibility  
- Hosting on **GCP Compute Engine** with proper firewall & networking setup  

Even on CPU, the API is operational and suitable for demo, testing, and prototyping (response time 1–3 minutes).

---

# 🧠 Model Details

| Component | Value |
|----------|-------|
| Base Model | `tiiuae/falcon-7b-instruct` |
| Fine-tuning | LoRA adapters (trained via Kaggle + Tinker workflow) |
| Serving Hardware | GCP Compute Engine (CPU only) |
| Format | PyTorch |
| Frameworks | transformers, peft, accelerate, torch |

---

# 🏗️ Architecture

Client → FastAPI → Docker Container → Falcon-7B Model + LoRA → CPU Inference → Response

---

# 📚 Training Workflow (Kaggle GPU)

1. Prepared dataset & notebook for LoRA fine-tuning  
2. Enabled **Kaggle GPU (T4)**  
3. Fine-tuned Falcon-7B using low-rank adapters  
4. Exported `lora-devotee/` directory  
5. Added LoRA weights to Docker image for deployment  

Kaggle was chosen because GCP VM did not provide GPU resources.

---

# ☁️ GCP Deployment

### VM Details
- OS: Ubuntu 22.04  
- Machine Type: e2-standard series  
- Port Exposed: **8000**  
- Authentication: SSH keys  
- Firewall: Custom ingress rule allowing TCP:8000  

---

# 🐳 Docker Deployment Instructions

```bash
docker build -t devotee-api .
docker run -d --name ai_chat_service -p 8000:8000 devotee-api
```

Check logs:

```bash
docker logs -f ai_chat_service
```

Stop/start service:

```bash
docker stop ai_chat_service
docker start ai_chat_service
```

---

# 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Rahul446K/gcp-ml-api-deployment.git
cd gcp-ml-api-deployment
```

### 2. Optional: Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run locally (without Docker)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

# 🔧 Environment Variables (Recommended)

Create `.env` file:

```
HOST=0.0.0.0
PORT=8000
MODEL_NAME=tiiuae/falcon-7b-instruct
LORA_DIR=./lora-devotee
DEVICE=cpu
MAX_TOKENS=120
TEMPERATURE=0.7
TOP_P=0.9
```

Add `.env` to `.gitignore`:

```
.env
```

---

# 🌐 Public API Base URL

```
http://34.60.125.198:8000
```

---

# 🔌 API Endpoints

## 1️⃣ Health Check – GET `/`

```
http://34.60.125.198:8000/
```

Response:
```json
{"message": "Devotee Chat API running!"}
```

---

## 2️⃣ Chat Inference – POST `/chat`

```
http://34.60.125.198:8000/chat
```

Request:
```json
{"prompt": "Hare Krishna, please guide me."}
```

Response:
```json
{"response": "Model generated text..."}
```

⚠️ Response time on CPU: **1–3 minutes**

---

## 3️⃣ API Documentation – `/docs`

```
http://34.60.125.198:8000/docs
```

---

# 🛑 Troubleshooting

| Issue | Reason | Fix |
|-------|--------|------|
| OOM error (Out Of Memory) | Model too large for RAM | Used accelerate offloading |
| Git auth failed | Password login deprecated | Use SSH key or PAT |
| Port unreachable | Firewall closed | Allow TCP:8000 |
| Docker build slow | Large model files | Expected (10–20 mins) |
| `/chat` not working in browser | Only accepts POST | Use curl/Postman |

---

# 📈 Future Enhancements

- Deploy on GPU VM for fast inference  
- Add request streaming  
- Add authentication & rate limits  
- Quantize model (4-bit GGUF)  
- Serve via vLLM or TGI for high throughput  

---

# 📄 License  
MIT License

---

# ✨ Author  
**Rahul Kumar**
