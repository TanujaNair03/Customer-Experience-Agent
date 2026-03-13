# Agentic CX Support Router

A portfolio project that demonstrates an agentic customer support workflow for an enterprise CX automation use case.

This app uses:

- `LangGraph` for workflow orchestration
- `LangChain` for retrieval and model integration
- `Google Gemini` for intent classification and answer generation
- `ChromaDB` for local FAQ retrieval
- `Streamlit` for a polished chat UI

The router decides whether a user message should:

- be answered using FAQ retrieval and generation
- be escalated to a human support agent

It also includes fallback logic so the demo remains usable when Gemini free-tier quota is exhausted.

## Project Structure

- [app.py](app.py): Streamlit UI
- [agent.py](agent.py): LangGraph routing workflow
- [data_setup.py](data_setup.py): dummy FAQ creation and Chroma indexing
- [test_agent.py](test_agent.py): pytest routing tests
- [data/sportswear_faq.txt](data/sportswear_faq.txt): dummy FAQ corpus

## Features

- Intent routing between FAQ support and human escalation
- Retrieval-augmented answers for returns, shipping, exchanges, damaged items, and international delivery
- Human-handoff payload with ticket metadata
- Guided quick-action buttons in the UI
- Custom black, white, and cream Streamlit interface
- Heuristic fallback mode for:
  - Gemini quota exhaustion
  - temporary API failures
  - common FAQ questions
  - compliment and negative-sentiment handling

## Demo Scenarios

Try messages like:

- `What is your return policy?`
- `How long does shipping take?`
- `Can I exchange an item for another size?`
- `Do you ship internationally?`
- `Can I get them in Argentina?`
- `terrible service`
- `horrible experience. never coming back here`
- `pretty products!`

## How It Works

### 1. Intent Router

The first node classifies the user message into:

- `policy_question`
- `angry_escalation`

Under normal conditions this uses Gemini. If Gemini quota is exhausted, the project falls back to local heuristics.

### 2. RAG Tool

For FAQ-style questions, the app:

- retrieves relevant chunks from ChromaDB
- prompts Gemini to answer using only retrieved context
- falls back to deterministic FAQ responses if the model or retrieval layer is unavailable

### 3. Human Escalation

For frustrated or urgent messages, the app skips retrieval and generates a structured escalation payload containing:

- ticket ID
- queue
- priority
- handoff summary
- customer message
- estimated response time

In a production system, this payload would be sent to a helpdesk platform such as Zendesk, Salesforce Service Cloud, or Freshdesk.

## Setup

Clone the repository and move into the project folder:

```bash
git clone <your-repo-url>
cd "Customer Experience Automation"
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-community langchain-google-genai langchain-chroma langchain-text-splitters chromadb pytest streamlit typing_extensions
```

Set your Gemini API key:

```bash
export GOOGLE_API_KEY="your_gemini_api_key"
```

## Build the Vector Store

Run:

```bash
python data_setup.py
```

This will:

- generate a small sportswear FAQ corpus
- chunk it using LangChain text splitters
- embed it using Google embeddings
- store it locally in `chroma_db/`

## Run the App

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## How To Use This Repo

Follow this order:

1. Install dependencies
2. Export `GOOGLE_API_KEY`
3. Run `python data_setup.py`
4. Run `streamlit run app.py`
5. Open the local Streamlit URL in your browser

## How To Use The App

Once the app is running, you can:

- type your own customer support question
- click one of the starter prompt buttons
- test escalation with an angry message
- test fallback support questions when Gemini quota is exhausted

Example prompts:

```text
What is your return policy?
How long does shipping take?
Can I get them in Argentina?
My order arrived damaged. What should I do?
terrible service
horrible experience. never coming back here
pretty products!
```

## Run Tests

```bash
pytest test_agent.py
```

## Notes on Gemini Quota

Gemini free-tier limits can be reached quickly during UI testing. This project is designed to degrade gracefully:

- negative or urgent messages still escalate
- common FAQ questions still receive fallback answers
- compliment-only messages still get a natural response

If you want full Gemini behavior again, wait for quota reset or use a higher-quota API plan.

## Why This Project Is Relevant

This project demonstrates:

- agentic workflow design
- LLM routing logic
- retrieval-augmented generation
- evaluation-minded testing
- human-in-the-loop escalation
- practical failure handling for real-world demos

## Future Improvements

- Send escalation payloads to a real ticketing backend
- Add observability for route choice and latency
- Add structured evaluation for hallucination and routing accuracy
- Expand the FAQ corpus and support order-status workflows
