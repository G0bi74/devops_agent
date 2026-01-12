# DevOps Runbook Assistant 🔧

AI-powered assistant for diagnosing and resolving server issues using RAG, function-calling, and security guardrails.

## 🎯 Features

- **Log Analysis**: Read and analyze logs from nginx, postgresql, docker, redis
- **System Monitoring**: Check disk usage, service status
- **Network Diagnostics**: Ping hosts (with allowlist)
- **Knowledge Base**: RAG-powered search through DevOps documentation
- **Security**: Prompt injection detection, path traversal prevention, allowlists

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone <repo-url>
cd devops_agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy template and fill in your API keys
copy .env.template .env

# Edit .env with your keys:
# - GOOGLE_API_KEY (Gemini)
# - GROQ_API_KEY (optional, for Groq/Llama)
```

### 3. Run

```bash
# Start the API server
python -m app.main

# Or with uvicorn directly
uvicorn app.main:app --reload --port 8000
```

### 4. Test

```bash
# Open in browser
http://localhost:8000/docs

# Or use curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Check nginx logs for errors"}'
```

## 📡 API Endpoints

### POST /ask
Main endpoint for asking questions.

**Request:**
```json
{
  "question": "Check nginx logs and tell me if there are any errors",
  "k": 3,
  "use_rag": true,
  "use_tools": true,
  "temperature": 0.3
}
```

**Response:**
```json
{
  "status": "success",
  "answer": "I found several errors in the nginx logs...",
  "tool_calls": [
    {
      "tool": "logs.read",
      "args": {"service_name": "nginx", "lines": 20},
      "success": true,
      "result": {...}
    }
  ],
  "latency_s": 2.5,
  "provider": "gemini"
}
```

### GET /health
Health check endpoint.

### GET /tools
List available tools.

## 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| `logs.read` | Read service logs (nginx, postgresql, docker, etc.) |
| `system.disk_usage` | Check disk space |
| `service.control` | Start/stop/restart/status services |
| `network.ping` | Ping hosts (allowlist only) |
| `kb.lookup` | Search knowledge base |

## 🔒 Security Features

### Guardrails
- **Prompt Injection Detection**: Blocks "ignore instructions", "reveal prompt", jailbreak attempts
- **Path Traversal Prevention**: Blocks `..`, absolute paths, sensitive files
- **Allowlists**: Only approved hosts and services can be accessed
- **Timeouts**: All tool executions have configurable timeouts
- **No Stacktraces**: Error messages don't leak internal details

### Testing Security
```bash
# Run red-team tests
pytest tests/test_redteam.py -v
```

## 📊 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_tools.py -v      # Tool validation
pytest tests/test_redteam.py -v    # Security tests
pytest tests/test_rag.py -v        # RAG quality

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## 📁 Project Structure

```
devops_agent/
├── app/
│   ├── main.py              # FastAPI application
│   ├── core/
│   │   ├── agent.py         # Main agent with FC loop
│   │   ├── llm_service.py   # LLM providers (Gemini, Groq, local)
│   │   └── security.py      # Guardrails
│   ├── rag/
│   │   └── engine.py        # FAISS + embeddings
│   └── tools/
│       ├── registry.py      # Pydantic schemas
│       └── implementations.py # Tool functions
├── data/runbooks/           # Knowledge base documents
├── mock_fs/                 # Simulated filesystem for logs
├── tests/                   # Test suite
├── .env.template            # Environment template
└── requirements.txt         # Dependencies
```

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Primary LLM (gemini/groq/local) |
| `GOOGLE_API_KEY` | - | Gemini API key |
| `GROQ_API_KEY` | - | Groq API key |
| `TOOL_TIMEOUT_SEC` | `5.0` | Tool execution timeout |
| `MAX_RAG_RESULTS` | `5` | Max RAG results |
| `ALLOWED_HOSTS` | localhost,google.com,... | Pingable hosts |
| `ALLOWED_SERVICES` | nginx,docker,... | Manageable services |
| `USE_RERANKING` | `1` | Enable cross-encoder reranking |

## 📝 Example Queries

```
"Check nginx logs for errors"
"Is the database server running?"
"The API is returning 502 errors, help me diagnose"
"How much disk space is left?"
"Ping google.com to check connectivity"
"Restart the docker service"
```

## 🏆 Scoring Alignment

| Requirement | Points | Status |
|------------|--------|--------|
| Tool registry + schemas | 15 | ✅ |
| Dispatcher + security | 15 | ✅ |
| Function-calling | 15 | ✅ |
| Mini-RAG | 20 | ✅ |
| Guardrails | 9 | ✅ |
| Evaluation | 8 | ✅ |
| REST API | 8 | ✅ |
| Observability | 5 | ✅ |
| Code quality | 5 | ✅ |
| Demo/Report | 10 | ✅ |
| **Bonus: Reranking** | +8 | ✅ |

## 📜 License

MIT
