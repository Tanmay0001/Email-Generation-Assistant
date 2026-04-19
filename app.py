from fastapi import FastAPI
from pydantic import BaseModel
from llm import generate_email, MODEL_A, MODEL_B
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Email Generation Assistant",
    version="0.1.0",
    description="Professional email generator using Groq + LangChain"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    intent: str
    facts: str
    tone: str
    model: str = "A"   # A or B

class EmailResponse(BaseModel):
    email: str
    model_used: str

@app.get("/health")
def health():
    return {"status": "healthy", "message": "Email Assistant is running"}

@app.post("/generate", response_model=EmailResponse)
def generate_email_endpoint(request: EmailRequest):
    try:
        model_id = MODEL_A if request.model.upper() == "A" else MODEL_B
        
        print(f"[INFO] Generating email with model: {model_id}")
        print(f"[INFO] Intent: {request.intent}")
        
        email = generate_email(
            intent=request.intent,
            facts=request.facts,
            tone=request.tone,
            model_id=model_id
        )
        
        return EmailResponse(
            email=email,
            model_used=model_id
        )
        
    except Exception as e:
        error_msg = f"Error generating email: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return EmailResponse(
            email=error_msg,
            model_used="error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)