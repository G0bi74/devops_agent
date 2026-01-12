"""
LLM Service - obsługa wielu providerów LLM.
Wspiera: Gemini, Groq (OpenAI-compatible), lokalny model.
"""
import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================
# Abstrakcyjna klasa bazowa dla providerów
# ============================================================

class LLMProvider(ABC):
    """Interfejs dla wszystkich providerów LLM."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Wywołanie LLM z opcjonalnym function-calling.
        
        Returns:
            {
                "content": str | None,
                "function_call": {"name": str, "arguments": dict} | None,
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
                "latency_s": float
            }
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Sprawdza czy provider jest skonfigurowany."""
        pass


# ============================================================
# Gemini Provider (Google AI)
# ============================================================

class GeminiProvider(LLMProvider):
    """Google Gemini z natywnym function-calling."""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini provider initialized: {self.model_name}")
            except ImportError:
                logger.warning("google-genai not installed")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def _convert_functions_to_gemini(self, functions: List[Dict]) -> List[Dict]:
        """Konwertuje OpenAI-style functions do formatu Gemini."""
        if not functions:
            return []
        
        tools = []
        for func in functions:
            tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}})
            }
            tools.append({"function_declarations": [tool]})
        return tools
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        from google import genai
        
        # Budowanie promptu z messages
        system_msg = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                user_content += f"\nAssistant: {msg['content']}"
            elif msg["role"] == "function":
                user_content += f"\nFunction result ({msg.get('name', 'tool')}): {msg['content']}"
        
        config = genai.types.GenerateContentConfig(
            system_instruction=system_msg or "You are a helpful DevOps assistant.",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Dodaj tools jeśli są
        if functions:
            tools = self._convert_functions_to_gemini(functions)
            config.tools = tools
        
        t0 = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=config,
            )
            latency = round(time.time() - t0, 3)
            
            # Parsowanie odpowiedzi
            result = {
                "content": None,
                "function_call": None,
                "usage": {},
                "latency_s": latency
            }
            
            # Sprawdź czy jest function call
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            result["function_call"] = {
                                "name": fc.name,
                                "arguments": dict(fc.args) if fc.args else {}
                            }
                        elif hasattr(part, 'text'):
                            result["content"] = part.text
            
            # Fallback na .text
            if not result["content"] and not result["function_call"]:
                result["content"] = getattr(response, 'text', str(response))
            
            # Usage
            usage = getattr(response, "usage_metadata", None)
            if usage:
                result["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                    "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0),
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return {
                "content": None,
                "function_call": None,
                "error": str(e),
                "latency_s": round(time.time() - t0, 3)
            }


# ============================================================
# Groq Provider (OpenAI-compatible, bardzo szybki)
# ============================================================

class GroqProvider(LLMProvider):
    """Groq - superszybkie LLM (Llama, Mixtral) z OpenAI-compatible API."""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                logger.info(f"Groq provider initialized: {self.model_name}")
            except ImportError:
                logger.warning("openai package not installed")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        t0 = time.time()
        
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Function calling przez tools (OpenAI format)
            if functions:
                kwargs["tools"] = [
                    {"type": "function", "function": f} for f in functions
                ]
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            latency = round(time.time() - t0, 3)
            
            message = response.choices[0].message
            
            result = {
                "content": message.content,
                "function_call": None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_s": latency
            }
            
            # Sprawdź tool_calls
            if message.tool_calls:
                tc = message.tool_calls[0]
                result["function_call"] = {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return {
                "content": None,
                "function_call": None,
                "error": str(e),
                "latency_s": round(time.time() - t0, 3)
            }


# ============================================================
# Local Provider (Transformers + GPU)
# ============================================================

class LocalProvider(LLMProvider):
    """Lokalny model (Qwen) z symulowanym function-calling."""
    
    def __init__(self):
        self.model_name = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def _load_model(self):
        """Lazy loading modelu."""
        if self.model is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading local model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        # Sprawdzamy czy torch jest zainstalowany
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        import torch
        
        self._load_model()
        
        if self.model is None:
            return {
                "content": None,
                "function_call": None,
                "error": "Model not loaded",
                "latency_s": 0
            }
        
        t0 = time.time()
        
        # Budowanie promptu z instrukcją dla function-calling
        if functions:
            # Dodajemy instrukcję FC do system message
            fc_instruction = self._build_fc_instruction(functions)
            enhanced_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    enhanced_messages.append({
                        "role": "system",
                        "content": msg["content"] + "\n\n" + fc_instruction
                    })
                else:
                    enhanced_messages.append(msg)
            if not any(m["role"] == "system" for m in enhanced_messages):
                enhanced_messages.insert(0, {"role": "system", "content": fc_instruction})
        else:
            enhanced_messages = messages
        
        try:
            # Generowanie
            if hasattr(self.tokenizer, "apply_chat_template"):
                model_inputs = self.tokenizer.apply_chat_template(
                    enhanced_messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True
                ).to(self.device)
            else:
                text = "\n".join([f"[{m['role'].upper()}]\n{m['content']}" for m in enhanced_messages])
                model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9
                
                output_ids = self.model.generate(**model_inputs, **gen_kwargs)
            
            # Dekodowanie
            if hasattr(self.tokenizer, "apply_chat_template"):
                gen_ids = output_ids[:, model_inputs["input_ids"].shape[-1]:]
            else:
                gen_ids = output_ids
            
            text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            latency = round(time.time() - t0, 3)
            
            result = {
                "content": text,
                "function_call": None,
                "usage": {
                    "prompt_tokens": model_inputs["input_ids"].shape[-1],
                    "completion_tokens": gen_ids.shape[-1],
                    "total_tokens": model_inputs["input_ids"].shape[-1] + gen_ids.shape[-1],
                },
                "latency_s": latency
            }
            
            # Próba parsowania function call z odpowiedzi
            if functions:
                fc = self._parse_function_call(text, functions)
                if fc:
                    result["function_call"] = fc
                    result["content"] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return {
                "content": None,
                "function_call": None,
                "error": str(e),
                "latency_s": round(time.time() - t0, 3)
            }
    
    def _build_fc_instruction(self, functions: List[Dict]) -> str:
        """Buduje instrukcję function-calling dla lokalnego modelu."""
        funcs_desc = []
        for f in functions:
            funcs_desc.append(f"- {f['name']}: {f.get('description', '')}")
            if "parameters" in f and "properties" in f["parameters"]:
                for param, spec in f["parameters"]["properties"].items():
                    funcs_desc.append(f"    - {param}: {spec.get('type', 'any')}")
        
        return f"""You have access to these tools:
{chr(10).join(funcs_desc)}

When you need to use a tool, respond ONLY with a JSON object in this exact format:
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}

If you don't need a tool, respond normally in plain text."""
    
    def _parse_function_call(self, text: str, functions: List[Dict]) -> Optional[Dict]:
        """Próbuje wyekstrahować function call z tekstu."""
        import re
        
        # Szukamy JSON w odpowiedzi
        json_patterns = [
            r'\{[^{}]*"tool"[^{}]*"args"[^{}]*\}',
            r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    # Normalizacja do naszego formatu
                    if "tool" in data:
                        return {"name": data["tool"], "arguments": data.get("args", {})}
                    elif "name" in data:
                        return {"name": data["name"], "arguments": data.get("arguments", {})}
                except json.JSONDecodeError:
                    continue
        
        return None


# ============================================================
# LLM Service - główny interfejs
# ============================================================

class LLMService:
    """
    Główny serwis LLM z automatycznym fallback.
    Kolejność: Gemini → Groq → Local
    """
    
    def __init__(self, preferred_provider: Optional[str] = None):
        self.providers: Dict[str, LLMProvider] = {
            "gemini": GeminiProvider(),
            "groq": GroqProvider(),
            "local": LocalProvider(),
        }
        
        # Ustal preferowanego providera
        preferred = preferred_provider or os.getenv("LLM_PROVIDER", "gemini")
        self.provider_order = [preferred]
        
        # Dodaj pozostałych jako fallback
        for name in ["gemini", "groq", "local"]:
            if name not in self.provider_order:
                self.provider_order.append(name)
        
        logger.info(f"LLM Service initialized. Provider order: {self.provider_order}")
    
    def get_active_provider(self) -> Optional[LLMProvider]:
        """Zwraca pierwszego dostępnego providera."""
        for name in self.provider_order:
            if self.providers[name].is_available():
                return self.providers[name]
        return None
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Wywołuje LLM z automatycznym fallback.
        """
        for name in self.provider_order:
            provider = self.providers[name]
            if not provider.is_available():
                continue
            
            logger.debug(f"Trying provider: {name}")
            result = provider.chat(
                messages=messages,
                functions=functions,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if "error" not in result:
                result["provider"] = name
                return result
            
            logger.warning(f"Provider {name} failed: {result.get('error')}")
        
        return {
            "content": None,
            "function_call": None,
            "error": "All providers failed",
            "latency_s": 0
        }


# ============================================================
# Singleton instance
# ============================================================

_llm_service: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    """Zwraca singleton LLM Service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    service = get_llm_service()
    
    # Test prosty
    result = service.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello DevOps!' in exactly 3 words."}
    ])
    
    print("=== Simple test ===")
    print(f"Provider: {result.get('provider')}")
    print(f"Content: {result.get('content')}")
    print(f"Latency: {result.get('latency_s')}s")
    
    # Test z function calling
    functions = [
        {
            "name": "read_logs",
            "description": "Read log files from a service",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Name of the service"},
                    "lines": {"type": "integer", "description": "Number of lines to read"}
                },
                "required": ["service_name"]
            }
        }
    ]
    
    result = service.chat(
        messages=[
            {"role": "system", "content": "You are a DevOps assistant. Use tools when needed."},
            {"role": "user", "content": "Check the last 20 lines of nginx logs"}
        ],
        functions=functions
    )
    
    print("\n=== Function calling test ===")
    print(f"Provider: {result.get('provider')}")
    print(f"Function call: {result.get('function_call')}")
    print(f"Content: {result.get('content')}")
