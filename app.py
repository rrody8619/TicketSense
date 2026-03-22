from fastapi import FastAPI
from pydantic import BaseModel
# تعديل الـ Imports عشان يقرأ من فولدر services
from services.ml import predict_ml
from services.nlp import analyse
from services.rag import rag_response

app = FastAPI()

class Ticket(BaseModel):
    title: str
    description: str

@app.get("/")
async def root():
    return {"message": "Welcome to TicketSense API! Go to /docs to test."}

@app.post("/ticket")
async def process_ticket(ticket: Ticket):
    full_text = f"{ticket.title} {ticket.description}"
    
    # استدعاء الدوال من الملفات المختلفة
    cat, prio = predict_ml(full_text)
    sentiment = analyse(full_text)
    solution = rag_response(full_text, cat, prio, sentiment)
    
    return {
        "category": cat,
        "priority": prio,
        "sentiment": sentiment,
        "suggested_solution": solution
    }