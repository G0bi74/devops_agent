"""
Tool Implementations - Implementacje narzędzi DevOps.

Punkty:
- (5 pkt) Timeout na tool
- (5 pkt) Sanitacja newralgicznych pól
- (5 pkt) Obsługa wyjątków i kategoryzacja
"""
import os
import time
import random
import logging
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, Any, Callable, Tuple, Optional
from datetime import datetime

from app.tools.registry import (
    ALLOWED_TOOLS,
    ReadLogsArgs,
    DiskUsageArgs,
    ServiceControlArgs,
    PingHostArgs,
    KBLookupArgs,
    validate_tool_args,
)
from app.core.security import (
    SecurityError,
    SecurityCategory,
    validate_host,
    validate_service,
    sanitize_path,
)

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

MOCK_FS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mock_fs"))
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT_SEC", "5.0"))


# ============================================================
# Tool Result Model
# ============================================================

class ToolResult:
    """Wynik wykonania narzędzia z kategoryzacją."""
    
    def __init__(
        self,
        success: bool,
        data: Dict[str, Any],
        error: Optional[str] = None,
        category: Optional[SecurityCategory] = None,
        execution_time: float = 0.0,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.category = category
        self.execution_time = execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "execution_time_ms": round(self.execution_time * 1000, 2),
        }
        if self.success:
            result["data"] = self.data
        else:
            result["error"] = self.error
            if self.category:
                result["error_category"] = self.category.value
        return result


# ============================================================
# Tool Implementations
# ============================================================

def _read_logs(args: ReadLogsArgs) -> Dict[str, Any]:
    """
    Czyta logi serwisu z mock filesystem.
    W produkcji: journalctl, /var/log, docker logs.
    """
    service = args.service_name
    lines = args.lines
    level = args.level or "all"
    
    # Mapowanie serwisu na plik logów
    log_files = {
        "nginx": "var/log/nginx/access.log",
        "postgresql": "var/log/postgresql/postgresql.log",
        "docker": "var/log/docker/docker.log",
        "redis": "var/log/redis/redis.log",
        "mysql": "var/log/mysql/mysql.log",
        "apache": "var/log/apache/access.log",
        "system": "var/log/syslog",
    }
    
    if service not in log_files:
        return {"error": f"Unknown service: {service}"}
    
    log_path = log_files[service]
    
    try:
        # Sanityzacja ścieżki (bezpieczeństwo!)
        full_path, _ = sanitize_path(log_path, MOCK_FS_ROOT)
        
        if not os.path.exists(full_path):
            # Generuj mock logi jeśli plik nie istnieje
            return _generate_mock_logs(service, lines, level)
        
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        
        # Filtrowanie po poziomie
        if level != "all":
            level_patterns = {
                "error": ["ERROR", "error", "FATAL", "fatal", "CRITICAL"],
                "warning": ["WARN", "warn", "WARNING", "warning"],
                "info": ["INFO", "info"],
            }
            patterns = level_patterns.get(level, [])
            all_lines = [l for l in all_lines if any(p in l for p in patterns)]
        
        # Ostatnie N linii
        selected_lines = all_lines[-lines:]
        
        return {
            "service": service,
            "lines_count": len(selected_lines),
            "level_filter": level,
            "content": "".join(selected_lines),
            "source": log_path,
        }
        
    except SecurityError as e:
        raise e
    except Exception as e:
        logger.error(f"Error reading logs for {service}: {e}")
        return {"error": str(e)}


def _generate_mock_logs(service: str, lines: int, level: str) -> Dict[str, Any]:
    """Generuje przykładowe logi dla demo."""
    log_templates = {
        "nginx": [
            "2025-01-12 10:00:{sec} [INFO] GET /api/health 200 0.005s",
            "2025-01-12 10:00:{sec} [INFO] POST /api/users 201 0.123s",
            "2025-01-12 10:00:{sec} [ERROR] GET /api/data 502 Bad Gateway",
            "2025-01-12 10:00:{sec} [WARNING] Connection timeout to upstream",
        ],
        "postgresql": [
            "2025-01-12 10:00:{sec} [INFO] connection authorized: user=app database=main",
            "2025-01-12 10:00:{sec} [ERROR] connection refused: too many clients",
            "2025-01-12 10:00:{sec} [WARNING] checkpoints are occurring too frequently",
            "2025-01-12 10:00:{sec} [INFO] autovacuum: processed database 'main'",
        ],
        "docker": [
            "2025-01-12 10:00:{sec} [INFO] Container web-1 started",
            "2025-01-12 10:00:{sec} [ERROR] Container db-1 exited with code 137 (OOM)",
            "2025-01-12 10:00:{sec} [WARNING] Low disk space on /var/lib/docker",
            "2025-01-12 10:00:{sec} [INFO] Image nginx:latest pulled successfully",
        ],
    }
    
    templates = log_templates.get(service, [
        "2025-01-12 10:00:{sec} [INFO] Service {service} running normally",
        "2025-01-12 10:00:{sec} [WARNING] High memory usage detected",
        "2025-01-12 10:00:{sec} [ERROR] Connection lost, retrying...",
    ])
    
    generated = []
    for i in range(lines):
        template = random.choice(templates)
        line = template.format(sec=str(i).zfill(2), service=service)
        
        # Filtruj po poziomie
        if level != "all":
            level_upper = level.upper()
            if level_upper not in line.upper():
                continue
        
        generated.append(line)
    
    return {
        "service": service,
        "lines_count": len(generated),
        "level_filter": level,
        "content": "\n".join(generated) if generated else "No matching log entries",
        "source": "[mock data]",
    }


def _check_disk_usage(args: DiskUsageArgs) -> Dict[str, Any]:
    """
    Sprawdza użycie dysku.
    W produkcji: df, psutil.
    """
    path = args.path
    
    try:
        # Dla demo - symulowane dane
        # W produkcji użyj: import shutil; shutil.disk_usage(path)
        
        # Symulacja realistycznych wartości
        total_gb = random.uniform(100, 500)
        used_percent = random.uniform(30, 85)
        used_gb = total_gb * (used_percent / 100)
        free_gb = total_gb - used_gb
        
        return {
            "path": path,
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "used_percent": round(used_percent, 1),
            "status": "critical" if used_percent > 90 else "warning" if used_percent > 80 else "ok",
        }
        
    except Exception as e:
        logger.error(f"Error checking disk usage: {e}")
        return {"error": str(e)}


def _service_control(args: ServiceControlArgs) -> Dict[str, Any]:
    """
    Zarządza serwisami systemowymi.
    W produkcji: systemctl, docker compose.
    """
    service = args.service_name
    action = args.action
    force = args.force
    
    # Walidacja serwisu przez allowlist
    try:
        validate_service(service)
    except SecurityError as e:
        raise e
    
    # Symulacja czasów wykonania
    action_times = {
        "status": 0.1,
        "start": 1.5,
        "stop": 1.0,
        "restart": 2.5,
    }
    
    # Symulacja opóźnienia
    exec_time = action_times.get(action, 0.5)
    time.sleep(exec_time * 0.3)  # 30% rzeczywistego czasu dla demo
    
    # Symulowane odpowiedzi
    if action == "status":
        statuses = ["running", "running", "running", "stopped"]  # Przeważnie running
        status = random.choice(statuses)
        return {
            "service": service,
            "action": action,
            "status": status,
            "uptime": "3d 14h 22m" if status == "running" else None,
            "pid": random.randint(1000, 65000) if status == "running" else None,
        }
    
    elif action in ["start", "stop", "restart"]:
        # Symulacja sukcesu/porażki
        success = random.random() > 0.1  # 90% sukcesu
        
        if success:
            return {
                "service": service,
                "action": action,
                "result": "success",
                "message": f"Service {service} {action}ed successfully",
            }
        else:
            return {
                "service": service,
                "action": action,
                "result": "failed",
                "error": f"Failed to {action} {service}: permission denied or service locked",
            }
    
    return {"error": f"Unknown action: {action}"}


def _ping_host(args: PingHostArgs) -> Dict[str, Any]:
    """
    Pinguje hosta (z allowlist!).
    """
    host = args.host
    count = args.count
    timeout = args.timeout
    
    # KRYTYCZNE: Walidacja przez allowlist
    try:
        validate_host(host)
    except SecurityError as e:
        raise e
    
    # Symulowane wyniki pinga (dla demo)
    # W produkcji można użyć subprocess z ping
    
    packets_sent = count
    packets_received = count if random.random() > 0.1 else count - 1
    packet_loss = ((packets_sent - packets_received) / packets_sent) * 100
    
    # Symulowane RTT
    rtts = [random.uniform(10, 100) for _ in range(packets_received)]
    
    return {
        "host": host,
        "packets_sent": packets_sent,
        "packets_received": packets_received,
        "packet_loss_percent": round(packet_loss, 1),
        "rtt_min_ms": round(min(rtts), 2) if rtts else None,
        "rtt_avg_ms": round(sum(rtts) / len(rtts), 2) if rtts else None,
        "rtt_max_ms": round(max(rtts), 2) if rtts else None,
        "status": "reachable" if packets_received > 0 else "unreachable",
    }


def _kb_lookup(args: KBLookupArgs) -> Dict[str, Any]:
    """
    Wyszukiwanie w bazie wiedzy.
    Właściwa implementacja w RAG engine.
    """
    # To jest stub - prawdziwa implementacja w app.rag.engine
    return {
        "query": args.query,
        "message": "KB lookup delegated to RAG engine",
        "top_k": args.top_k,
    }


# ============================================================
# Tool Mapping (Dispatcher)
# ============================================================

TOOL_IMPLEMENTATIONS: Dict[str, Callable] = {
    "logs.read": _read_logs,
    "system.disk_usage": _check_disk_usage,
    "service.control": _service_control,
    "network.ping": _ping_host,
    "kb.lookup": _kb_lookup,
}


# ============================================================
# Main Dispatcher (z timeout i obsługą błędów)
# ============================================================

def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    timeout: Optional[float] = None,
) -> ToolResult:
    """
    Wykonuje narzędzie z timeoutem i pełną obsługą błędów.
    
    Args:
        tool_name: Nazwa narzędzia z ALLOWED_TOOLS
        args: Argumenty narzędzia
        timeout: Timeout w sekundach (domyślnie z TOOL_TIMEOUT)
    
    Returns:
        ToolResult z kategoryzacją błędów
    """
    timeout = timeout or TOOL_TIMEOUT
    t0 = time.time()
    
    # Krok 1: Sprawdź czy narzędzie jest dozwolone
    if tool_name not in ALLOWED_TOOLS:
        return ToolResult(
            success=False,
            data={},
            error=f"Tool '{tool_name}' is not allowed",
            category=SecurityCategory.VALIDATION_ERROR,
            execution_time=time.time() - t0,
        )
    
    # Krok 2: Walidacja argumentów przez Pydantic
    try:
        validated_args = validate_tool_args(tool_name, args)
    except ValueError as e:
        return ToolResult(
            success=False,
            data={},
            error=str(e),
            category=SecurityCategory.VALIDATION_ERROR,
            execution_time=time.time() - t0,
        )
    
    # Krok 3: Pobierz implementację
    impl_func = TOOL_IMPLEMENTATIONS.get(tool_name)
    if impl_func is None:
        return ToolResult(
            success=False,
            data={},
            error=f"No implementation for tool: {tool_name}",
            category=SecurityCategory.TOOL_ERROR,
            execution_time=time.time() - t0,
        )
    
    # Krok 4: Wykonaj z timeoutem
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(impl_func, validated_args)
        
        try:
            result = future.result(timeout=timeout)
            exec_time = time.time() - t0
            
            # Sprawdź czy wynik zawiera błąd
            if isinstance(result, dict) and "error" in result:
                return ToolResult(
                    success=False,
                    data={},
                    error=result["error"],
                    category=SecurityCategory.TOOL_ERROR,
                    execution_time=exec_time,
                )
            
            # Logowanie sukcesu
            logger.info(f"Tool {tool_name} executed in {exec_time:.3f}s")
            
            return ToolResult(
                success=True,
                data=result,
                execution_time=exec_time,
            )
            
        except FuturesTimeout:
            logger.warning(f"Tool {tool_name} timed out after {timeout}s")
            return ToolResult(
                success=False,
                data={},
                error=f"Tool execution timed out after {timeout}s",
                category=SecurityCategory.TIMEOUT,
                execution_time=timeout,
            )
            
        except SecurityError as e:
            logger.warning(f"Security error in {tool_name}: {e.message}")
            return ToolResult(
                success=False,
                data={},
                error=e.message,
                category=e.category,
                execution_time=time.time() - t0,
            )
            
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}")
            # NIE zwracamy stacktrace do klienta!
            return ToolResult(
                success=False,
                data={},
                error=f"Tool execution failed: {type(e).__name__}",
                category=SecurityCategory.TOOL_ERROR,
                execution_time=time.time() - t0,
            )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Tools ===\n")
    
    # Test 1: Read logs
    print("1. Read nginx logs:")
    result = execute_tool("logs.read", {"service_name": "nginx", "lines": 5})
    print(result.to_dict())
    
    # Test 2: Disk usage
    print("\n2. Check disk usage:")
    result = execute_tool("system.disk_usage", {"path": "/"})
    print(result.to_dict())
    
    # Test 3: Service status
    print("\n3. Service status:")
    result = execute_tool("service.control", {"service_name": "docker", "action": "status"})
    print(result.to_dict())
    
    # Test 4: Ping allowed host
    print("\n4. Ping allowed host:")
    result = execute_tool("network.ping", {"host": "google.com", "count": 3})
    print(result.to_dict())
    
    # Test 5: Ping blocked host
    print("\n5. Ping blocked host (should fail):")
    result = execute_tool("network.ping", {"host": "evil.com", "count": 3})
    print(result.to_dict())
    
    # Test 6: Invalid tool
    print("\n6. Invalid tool:")
    result = execute_tool("rm.rf", {"path": "/"})
    print(result.to_dict())
    
    # Test 7: Invalid arguments
    print("\n7. Invalid arguments:")
    result = execute_tool("logs.read", {"service_name": "unknown_service"})
    print(result.to_dict())
