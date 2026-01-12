"""
Testy narzędzi - walidacja, bezpieczeństwo, timeouty.

Przypadki testowe:
- Format/walidacja argumentów
- Bezpieczeństwo (path traversal, allowlist)
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.registry import (
    ALLOWED_TOOLS,
    ReadLogsArgs,
    PingHostArgs,
    ServiceControlArgs,
    validate_tool_args,
)
from app.tools.implementations import execute_tool
from app.core.security import SecurityCategory


class TestToolValidation:
    """Testy walidacji argumentów (2 przypadki format/JSON)."""
    
    def test_valid_read_logs_args(self):
        """Test poprawnych argumentów dla logs.read."""
        args = ReadLogsArgs(service_name="nginx", lines=50, level="error")
        assert args.service_name == "nginx"
        assert args.lines == 50
        assert args.level == "error"
    
    def test_invalid_service_name(self):
        """Test nieprawidłowej nazwy serwisu."""
        with pytest.raises(ValueError):
            ReadLogsArgs(service_name="unknown_service", lines=10)
    
    def test_lines_out_of_range(self):
        """Test linii poza zakresem."""
        with pytest.raises(ValueError):
            ReadLogsArgs(service_name="nginx", lines=500)  # max 100
    
    def test_valid_ping_args(self):
        """Test poprawnych argumentów dla ping."""
        args = PingHostArgs(host="google.com", count=5, timeout=10)
        assert args.host == "google.com"
        assert args.count == 5
    
    def test_ping_command_injection_blocked(self):
        """Test blokady command injection w host."""
        with pytest.raises(ValueError):
            PingHostArgs(host="google.com; rm -rf /", count=4)
    
    def test_validate_tool_args_function(self):
        """Test funkcji validate_tool_args."""
        validated = validate_tool_args("logs.read", {"service_name": "docker", "lines": 20})
        assert validated.service_name == "docker"
        assert validated.lines == 20
    
    def test_validate_unknown_tool(self):
        """Test nieznanego narzędzia."""
        with pytest.raises(ValueError, match="Unknown tool"):
            validate_tool_args("unknown.tool", {})


class TestToolExecution:
    """Testy wykonania narzędzi."""
    
    def test_execute_allowed_tool(self):
        """Test wykonania dozwolonego narzędzia."""
        result = execute_tool("logs.read", {"service_name": "nginx", "lines": 5})
        assert result.success or "error" in result.to_dict()
    
    def test_execute_forbidden_tool(self):
        """Test blokady niedozwolonego narzędzia."""
        result = execute_tool("rm.rf", {"path": "/"})
        assert not result.success
        assert result.category == SecurityCategory.VALIDATION_ERROR
    
    def test_disk_usage_tool(self):
        """Test narzędzia disk_usage."""
        result = execute_tool("system.disk_usage", {"path": "/"})
        assert result.success
        assert "used_percent" in result.data
    
    def test_service_status(self):
        """Test sprawdzania statusu serwisu."""
        result = execute_tool("service.control", {
            "service_name": "nginx",
            "action": "status"
        })
        assert result.success
        assert "status" in result.data


class TestToolTimeout:
    """Testy timeoutów."""
    
    def test_tool_respects_timeout(self):
        """Test czy narzędzie respektuje timeout."""
        # Krótki timeout
        result = execute_tool(
            "service.control",
            {"service_name": "docker", "action": "restart"},
            timeout=0.1  # Bardzo krótki
        )
        # Może się udać lub timeout
        assert result.execution_time <= 1.0  # Nie powinno trwać długo


class TestAllowedTools:
    """Testy listy dozwolonych narzędzi."""
    
    def test_allowed_tools_defined(self):
        """Test czy lista dozwolonych narzędzi jest zdefiniowana."""
        assert len(ALLOWED_TOOLS) >= 4
        assert "logs.read" in ALLOWED_TOOLS
        assert "network.ping" in ALLOWED_TOOLS
    
    def test_all_tools_have_implementations(self):
        """Test czy wszystkie narzędzia mają implementacje."""
        from app.tools.implementations import TOOL_IMPLEMENTATIONS
        
        for tool in ALLOWED_TOOLS:
            if tool != "kb.lookup":  # KB lookup jest w RAG
                assert tool in TOOL_IMPLEMENTATIONS, f"Missing implementation for {tool}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
