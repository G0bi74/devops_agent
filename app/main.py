"""
DevOps Runbook Assistant - FastAPI Application.

REST API z endpointem /ask.

Punkty:
- (6 pkt) FastAPI endpoint /ask z parametrami
- (2 pkt) Instrukcja uruchomienia
"""
import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment before imports
load_dotenv()

from app.core.agent import get_agent, DevOpsAgent, AgentStatus
from app.core.security import detect_injection, SecurityCategory

# ============================================================
# Logging Setup
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Lifespan (startup/shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicjalizacja przy starcie."""
    logger.info("Starting DevOps Runbook Assistant...")
    
    # Inicjalizacja agenta
    agent = get_agent()
    
    # Próba zainicjalizowania RAG (opcjonalne)
    try:
        from app.rag.engine import get_rag_engine
        rag = get_rag_engine()
        agent.set_rag_engine(rag)
        logger.info("RAG engine initialized")
    except Exception as e:
        logger.warning(f"RAG engine not available: {e}")
    
    logger.info("DevOps Runbook Assistant ready!")
    
    yield
    
    logger.info("Shutting down...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="DevOps Runbook Assistant",
    description="""
AI-powered assistant for diagnosing and resolving server issues.

## Features
- **Log Analysis**: Read and analyze service logs
- **System Monitoring**: Check disk usage, service status
- **Network Diagnostics**: Ping hosts
- **Knowledge Base**: Search DevOps documentation

## Security
- Prompt injection detection
- Path traversal prevention
- Allowlist for hosts and services
- Timeout on all tool executions
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (dla frontendu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class AskRequest(BaseModel):
    """Request model dla /ask endpoint."""
    
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Question or command for the assistant",
        examples=["Check nginx logs for errors", "Is the database server running?"]
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of RAG results to retrieve"
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use the knowledge base"
    )
    use_tools: bool = Field(
        default=True,
        description="Whether to use function calling (tools)"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0 = deterministic, 1 = creative)"
    )


class AskResponse(BaseModel):
    """Response model dla /ask endpoint."""
    
    status: str = Field(..., description="Response status: success, blocked, or error")
    answer: Optional[str] = Field(None, description="Assistant's answer")
    tool_calls: Optional[list] = Field(None, description="List of tool calls made")
    rag_context_used: Optional[bool] = Field(None, description="Whether RAG context was used")
    error: Optional[str] = Field(None, description="Error message if status is not success")
    error_category: Optional[str] = Field(None, description="Category of error")
    latency_s: float = Field(..., description="Total response time in seconds")
    usage: Optional[dict] = Field(None, description="Token usage statistics")
    provider: Optional[str] = Field(None, description="LLM provider used")


class HealthResponse(BaseModel):
    """Response model dla /health endpoint."""
    
    status: str
    version: str
    rag_available: bool
    llm_provider: Optional[str]


# ============================================================
# Endpoints
# ============================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "DevOps Runbook Assistant API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and availability of components.
    """
    agent = get_agent()
    
    rag_available = agent.rag_engine is not None
    
    llm_provider = None
    try:
        provider = agent.llm.get_active_provider()
        if provider:
            llm_provider = type(provider).__name__.replace("Provider", "")
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        rag_available=rag_available,
        llm_provider=llm_provider,
    )


@app.post("/ask", response_model=AskResponse, tags=["Assistant"])
async def ask(request: AskRequest):
    """
    Main endpoint - ask the DevOps assistant a question.
    
    The assistant can:
    - Read service logs (nginx, postgresql, docker, etc.)
    - Check disk usage
    - Control services (status, restart, stop)
    - Ping hosts (from allowlist only)
    - Search the knowledge base
    
    **Security**: Prompt injection attempts are blocked with HTTP 400.
    
    ## Examples
    
    ```bash
    curl -X POST http://localhost:8000/ask \\
      -H "Content-Type: application/json" \\
      -d '{"question": "Check nginx logs for errors"}'
    ```
    """
    agent = get_agent()
    
    # Quick injection check for 400 response
    is_injection, _ = detect_injection(request.question)
    if is_injection:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Request blocked due to security policy",
                "error_category": SecurityCategory.INJECTION_DETECTED.value,
            }
        )
    
    # Process request
    response = agent.ask(
        question=request.question,
        use_rag=request.use_rag,
        use_tools=request.use_tools,
        rag_top_k=request.k,
        temperature=request.temperature,
    )
    
    # Convert to response model
    result = response.to_dict()
    
    # Return appropriate status code
    if response.status == AgentStatus.BLOCKED:
        raise HTTPException(status_code=400, detail=result)
    elif response.status == AgentStatus.ERROR:
        raise HTTPException(status_code=500, detail=result)
    
    return result


@app.get("/tools", tags=["System"])
async def list_tools():
    """
    List available tools.
    
    Returns the list of tools the assistant can use with their schemas.
    """
    from app.tools.registry import TOOL_DEFINITIONS
    
    return {
        "tools": TOOL_DEFINITIONS,
        "count": len(TOOL_DEFINITIONS),
    }


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler - nie zwraca stacktrace do klienta."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "error_category": "internal_error",
        }
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           DevOps Runbook Assistant                            ║
║                                                               ║
║   API:     http://localhost:{port}                              ║
║   Docs:    http://localhost:{port}/docs                         ║
║   Health:  http://localhost:{port}/health                       ║
║                                                               ║
║   Press Ctrl+C to stop                                        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
