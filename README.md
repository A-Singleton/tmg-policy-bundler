# TMG Dynamic Policy Bundler Prototype

A high-velocity, production-ready prototype demonstrating dynamic insurance policy bundling via Retrieval-Augmented Generation (RAG). 

This application bridges modern Generative AI capabilities with strict, deterministic software engineering guardrails required for the highly regulated insurance industry. It evaluates a user's intent, queries mutual insurance documents, and proposes coverage bundles safely.

## 🏗 Architectural Philosophy & Key Decisions

This prototype was designed to simulate a robust 2-week sprint deliverable, prioritizing system integrity, latency optimization, and graceful degradation.

* **Decoupled Embedding Architecture (Zero-Latency RAG):** 
  To eliminate reliance on external API regions and reduce network latency during ingestion, the embedding model (`all-MiniLM-L6-v2` via HuggingFace) runs completely locally within the container memory. 
* **State-of-the-Art Inference (Gemini 2.5 Flash):** 
  The LLM orchestrator utilizes Google's active `gemini-2.5-flash` endpoint, optimized for rapid reasoning. The model temperature is strictly locked to `0.0` to enforce a deterministic matching role over creative generation.
* **Deterministic SWE Guardrails (`TMG_Guardrails`):** 
  The LLM is never granted direct write-access to the application state. AI outputs are parsed via Regex and routed through a strict Python middleware class. If the AI hallucinates a non-approved product or an illogical premium, the transaction is intercepted, blocked, and flagged for manual agent review.
* **Graceful UI Degradation:** 
  The interface offers traditional, deterministic UI routing buttons alongside the AI chat layer, ensuring continuous functionality regardless of user preference or AI confidence thresholds.

## ⚙️ Tech Stack
* **Frontend / State Management:** Streamlit
* **AI Orchestration:** LangChain (`langchain-google-genai`, `langchain-community`)
* **Vector Store:** FAISS (In-Memory CPU)
* **Embeddings:** Sentence Transformers / HuggingFace
* **Inference Engine:** Google Gemini 2.5 Flash

## 🚀 Local Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/tmg-policy-bundler.git](https://github.com/YOUR_USERNAME/tmg-policy-bundler.git)
cd tmg-policy-bundler
```

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
You will need a Google AI Studio API key. Create a `.streamlit/secrets.toml` file in the root directory (this is already ignored in `.gitignore`) and add your key:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_actual_api_key_here"
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will boot and be available locally at `http://localhost:8501`.

## 🧪 Testing the Guardrails
To observe the deterministic middleware in action, attempt to ask the AI to "bundle a life insurance policy for free." The `TMG_Guardrails` middleware will intercept the out-of-bounds request, blocking the application state change and safely routing the user to a human agent fallback.
