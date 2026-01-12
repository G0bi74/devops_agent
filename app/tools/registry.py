"""
Tool Registry - Schematy Pydantic + definicje narzędzi.

Punkty:
- (5 pkt) Lista dozwolonych narzędzi + rozdzielenie nazwa -> implementacja
- (6 pkt) Schematy argumentów (JSON Schema / Pydantic)
- (4 pkt) Obsługa walidacji błędów
"""
import os
from typing import Literal, Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ============================================================
# Allowed Tools (Allowlist) - 5 pkt
# ============================================================

class ToolName(str, Enum):
    """Lista dozwolonych narzędzi."""
    READ_LOGS = "logs.read"
    CHECK_DISK = "system.disk_usage"
    SERVICE_CONTROL = "service.control"
    PING_HOST = "network.ping"
    KB_LOOKUP = "kb.lookup"


ALLOWED_TOOLS = {tool.value for tool in ToolName}


# ============================================================
# Argument Schemas (Pydantic) - 6 pkt za typy, zakresy, enumy, limity
# ============================================================

class ReadLogsArgs(BaseModel):
    """Argumenty dla logs.read - czytanie logów serwisu."""
    
    service_name: Literal["nginx", "postgresql", "docker", "redis", "mysql", "apache", "system"] = Field(
        ...,
        description="Name of the service to read logs from (enum - only allowed services)"
    )
    lines: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of lines to read (1-100)"
    )
    level: Optional[Literal["all", "error", "warning", "info"]] = Field(
        default="all",
        description="Filter by log level"
    )
    
    @field_validator('service_name')
    @classmethod
    def validate_service(cls, v):
        # Dodatkowa walidacja - tylko dozwolone serwisy
        allowed = {"nginx", "postgresql", "docker", "redis", "mysql", "apache", "system"}
        if v.lower() not in allowed:
            raise ValueError(f"Service must be one of: {', '.join(sorted(allowed))}")
        return v.lower()


class DiskUsageArgs(BaseModel):
    """Argumenty dla system.disk_usage - sprawdzanie miejsca na dysku."""
    
    path: str = Field(
        default="/",
        max_length=128,
        description="Path to check disk usage for"
    )
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        # Blokada path traversal
        if ".." in v:
            raise ValueError("Path cannot contain '..'")
        return v


class ServiceControlArgs(BaseModel):
    """Argumenty dla service.control - zarządzanie serwisami."""
    
    service_name: Literal["nginx", "postgresql", "docker", "redis", "mysql", "apache"] = Field(
        ...,
        description="Name of the service to control"
    )
    action: Literal["status", "restart", "stop", "start"] = Field(
        ...,
        description="Action to perform on the service"
    )
    force: bool = Field(
        default=False,
        description="Force the action (skip confirmation for dangerous operations)"
    )


class PingHostArgs(BaseModel):
    """Argumenty dla network.ping - pingowanie hosta."""
    
    host: str = Field(
        ...,
        max_length=253,  # Max długość hostname
        description="Host to ping (must be in allowlist)"
    )
    count: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of ping packets (1-10)"
    )
    timeout: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Timeout in seconds (1-30)"
    )
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v):
        # Podstawowa walidacja - szczegółowa w security.py
        if not v or len(v.strip()) == 0:
            raise ValueError("Host cannot be empty")
        # Blokada oczywistych prób injection
        dangerous = [";", "|", "&", "`", "$", "(", ")", "{", "}", "<", ">"]
        for char in dangerous:
            if char in v:
                raise ValueError(f"Invalid character in host: {char}")
        return v.strip().lower()


class KBLookupArgs(BaseModel):
    """Argumenty dla kb.lookup - wyszukiwanie w bazie wiedzy."""
    
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Search query for knowledge base"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of results to return (1-10)"
    )


# ============================================================
# Tool Call Model
# ============================================================

class ToolCall(BaseModel):
    """Model wywołania narzędzia."""
    
    tool: str = Field(..., description="Tool name from allowed list")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    
    @field_validator('tool')
    @classmethod
    def validate_tool_name(cls, v):
        if v not in ALLOWED_TOOLS:
            raise ValueError(
                f"Tool '{v}' not allowed. Available tools: {', '.join(sorted(ALLOWED_TOOLS))}"
            )
        return v


# ============================================================
# Tool Definitions (dla LLM function-calling)
# ============================================================

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "logs.read",
        "description": "Read log files from a specified service. Use this to diagnose issues by checking recent log entries.",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "enum": ["nginx", "postgresql", "docker", "redis", "mysql", "apache", "system"],
                    "description": "Name of the service to read logs from"
                },
                "lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                    "description": "Number of log lines to retrieve"
                },
                "level": {
                    "type": "string",
                    "enum": ["all", "error", "warning", "info"],
                    "default": "all",
                    "description": "Filter logs by severity level"
                }
            },
            "required": ["service_name"]
        }
    },
    {
        "name": "system.disk_usage",
        "description": "Check disk usage for a given path. Returns used/available space and percentage.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "default": "/",
                    "description": "Path to check disk usage for"
                }
            },
            "required": []
        }
    },
    {
        "name": "service.control",
        "description": "Control system services (start, stop, restart, status). Use with caution for stop/restart.",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "enum": ["nginx", "postgresql", "docker", "redis", "mysql", "apache"],
                    "description": "Name of the service to control"
                },
                "action": {
                    "type": "string",
                    "enum": ["status", "restart", "stop", "start"],
                    "description": "Action to perform"
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force the action (for dangerous operations)"
                }
            },
            "required": ["service_name", "action"]
        }
    },
    {
        "name": "network.ping",
        "description": "Ping a host to check network connectivity. Only allowed hosts can be pinged.",
        "parameters": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Hostname or IP to ping (must be in allowlist)"
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 4,
                    "description": "Number of ping packets to send"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 5,
                    "description": "Timeout in seconds"
                }
            },
            "required": ["host"]
        }
    },
    {
        "name": "kb.lookup",
        "description": "Search the DevOps knowledge base for troubleshooting guides, documentation, and best practices.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 500,
                    "description": "Search query"
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    }
]


# ============================================================
# Schema Mapping
# ============================================================

ARGS_SCHEMA_MAP: Dict[str, type] = {
    "logs.read": ReadLogsArgs,
    "system.disk_usage": DiskUsageArgs,
    "service.control": ServiceControlArgs,
    "network.ping": PingHostArgs,
    "kb.lookup": KBLookupArgs,
}


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> BaseModel:
    """
    Waliduje argumenty narzędzia przy użyciu odpowiedniego schematu Pydantic.
    
    Raises:
        ValueError: jeśli narzędzie nieznane lub argumenty nieprawidłowe
    """
    if tool_name not in ARGS_SCHEMA_MAP:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    schema_class = ARGS_SCHEMA_MAP[tool_name]
    
    try:
        return schema_class(**args)
    except Exception as e:
        # Czytelny komunikat bez stacktrace (4 pkt za obsługę błędów)
        raise ValueError(f"Invalid arguments for {tool_name}: {str(e)}")


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Zwraca JSON Schema dla narzędzia."""
    for tool in TOOL_DEFINITIONS:
        if tool["name"] == tool_name:
            return tool
    return None
