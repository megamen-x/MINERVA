from fastapi import FastAPI
import re
from contextlib import asynccontextmanager
from pydantic import BaseModel
from classificators import seed_everything, inference_classificators
import subprocess


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_everything(seed=42)
    yield

lifespan=lifespan
app = FastAPI(title='Assistant API', version='0.1.0')

def remove_pattern(text):
    # Регулярное выражение для поиска паттерна
    pattern = r'\[.*\(\d+\)\]'    
    # Удаление всех совпадений паттерна из текста
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def remove_text_before_phrase(text, phrase):
    pattern = re.escape(phrase)
    result = re.sub(f'^.*?{pattern}', '', text, flags=re.DOTALL)
    return result.strip()

class InputText(BaseModel):
    question: str


@app.post("/send/")
async def send_response(input: InputText):
    class_1, class_2 = inference_classificators(input.question)
    proc = subprocess.Popen([f"""python3 -m graphrag.query --root ./raghacaton "{input.question}" """], stdout=subprocess.PIPE, shell=True)
    (out, _) = proc.communicate()
    out = out.decode('utf-8')
    out = remove_pattern(out)
    update_text = remove_text_before_phrase(out, 'Global Search Response:')
    return {
        'answer': update_text,
        'class_1': class_1,
        'class_2': class_2
    }