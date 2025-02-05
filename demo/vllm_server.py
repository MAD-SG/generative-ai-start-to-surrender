from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# VLLM server URL
VLLM_URL = "http://localhost:8000/v1/completions"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    try:
        payload = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "prompt": message,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "n": 1,
            "stream": False,
            "stop": None
        }

        response = requests.post(VLLM_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        return {"response": result["choices"][0]["text"].strip()}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30512)
