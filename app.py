from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Load a lightweight model
# generator = pipeline("text-generation", model="distilgpt2")

#generator = pipeline(
#    "text-generation",
#    model="tiiuae/falcon-rw-1b",
#    trust_remote_code=True
#)

generator = pipeline("text2text-generation", model="google/flan-t5-base")
model_name = "flan-t5-base"




class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(prompt: Prompt):
    result = generator(prompt.prompt, max_length=50)
    response_text = result[0]["generated_text"]
    return {"response": f"[{model_name}] {response_text.strip()}"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
