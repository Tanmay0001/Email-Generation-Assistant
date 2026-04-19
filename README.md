# Email Generation Assistant

AI powered professional email generator using Groq LLMs, LangChain, FastAPI and Streamlit.
A professional email generation assistant built using Groq LLM and LangChain. 
The system takes **Intent**, **Key Facts** and **Tone** as input and generates high quality business emails.

## Project Structure :

email-assistant/
├── app.py              # FastAPI backend — /generate and /health endpoints
├── llm.py              # Prompt engineering + Groq/LangChain inference
├── evaluation.py       # 3 custom metrics + model comparison runner
├── streamlit_app.py    # Minimal demo UI using stremlit
├── test_data.json      # 10 evaluation scenarios with human reference emails
├── results.csv         # Auto generated evaluation output
├── requirements.txt
└── .env                


## Features
- Advanced prompting (Role playing + Few shot examples)
- Support for two models: 
  - Model A: llama-3.3-70b-versatile (High Quality)
  - Model B: llama-3.1-8b-instant (Fast)
- Streamlit UI for easy interaction
- Custom evaluation framework with 3 metrics
- Model comparison on 10 test scenarios

## Tech Stack
- Backend: FastAPI + LangChain + Groq
- Frontend: Streamlit
- Evaluation: Custom Python script

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Add your Groq API key in .env : GROQ_API_KEY="................"

3. Start Backend: python -m uvicorn app:app --reload

4. Start UI (in new terminal): streamlit run streamlit_app.py


## Evaluation

10 test scenarios with reference emails (test_data.json)

3 Custom Metrics:

- Fact Recall: Measures how many key facts are included
- Tone Alignment: LLM as a Judge scoring (0.0–1.0)
- Conciseness: Penalizes overly long or short emails
- Results saved in results.csv



## Models

| Label   | Model ID                  | Role       |
|---------|---------------------------|------------|
| Model A | llama-3.3-70b-versatile   | Primary    |
| Model B | llama-3.1-8b-instant      | Comparison |

## Custom Metrics

| Metric         | Method              | Description                                      |
|----------------|---------------------|--------------------------------------------------|
| Fact Recall    | Keyword matching    | % of input facts present in generated email      |
| Tone Alignment | LLM-as-judge        | How well the email matches the requested tone    |
| Conciseness    | Word count scoring  | Penalises emails too short or too long           |

## Evaluation Results

| Model   | Fact Recall | Tone Alignment | Conciseness | Average |
|---------|-------------|----------------|-------------|---------|
| Model A | 0.750       | 0.900          | 0.912       | 0.854   |
| Model B | 0.900       | 0.890          | 0.933       | 0.908   |

## Prompting Technique

Uses a combination of **Role Playing** (Alexandra persona), **Few-Shot Examples**
(2 complete email examples) and **Chain-of-Thought** (4 step internal reasoning
process) to maximise output quality and consistency.


