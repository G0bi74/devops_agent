# DevOps Runbook Assistant 🔧

Asystent AI do diagnozowania i rozwiązywania problemów serwerowych z wykorzystaniem RAG, function-calling oraz zabezpieczeń (guardrails).

## 🎯 Funkcjonalności

- **Analiza logów**: Odczyt i analiza logów z nginx, postgresql, docker, redis
- **Monitoring systemu**: Sprawdzanie miejsca na dysku, statusu usług
- **Diagnostyka sieci**: Ping hostów (z białą listą)
- **Baza wiedzy**: Wyszukiwanie RAG w dokumentacji DevOps
- **Bezpieczeństwo**: Wykrywanie prompt injection, ochrona przed path traversal, białe listy

## 🚀 Szybki start

### 1. Instalacja

```bash
# Sklonuj repozytorium
git clone <repo-url>
cd devops_agent

# Utwórz wirtualne środowisko
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Zainstaluj zależności
pip install -r requirements.txt
```

### 2. Konfiguracja

```bash
# Skopiuj szablon i uzupełnij klucze API
copy .env.template .env

# Edytuj .env z kluczami:
# - GOOGLE_API_KEY (Gemini)
# - GROQ_API_KEY (opcjonalnie, dla Groq/Llama)
```

### 3. Uruchomienie

```bash
# Uruchom serwer API
python -m app.main

# Lub bezpośrednio przez uvicorn
uvicorn app.main:app --reload --port 8000
```

### 4. Testowanie

```bash
# Otwórz w przeglądarce
http://localhost:8000/docs

# Lub użyj curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Sprawdź logi nginx pod kątem błędów"}'
```

## 📡 Endpointy API

### POST /ask
Główny endpoint do zadawania pytań.

**Żądanie:**
```json
{
  "question": "Sprawdź logi nginx i powiedz czy są jakieś błędy",
  "k": 3,
  "use_rag": true,
  "use_tools": true,
  "temperature": 0.3
}
```

**Odpowiedź:**
```json
{
  "status": "success",
  "answer": "Znalazłem kilka błędów w logach nginx...",
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
Endpoint sprawdzający stan serwera.

### GET /tools
Lista dostępnych narzędzi.

## 🛠️ Dostępne narzędzia

| Narzędzie | Opis |
|-----------|------|
| `logs.read` | Odczyt logów usług (nginx, postgresql, docker, itp.) |
| `system.disk_usage` | Sprawdzanie miejsca na dysku |
| `service.control` | Start/stop/restart/status usług |
| `network.ping` | Ping hostów (tylko z białej listy) |
| `kb.lookup` | Wyszukiwanie w bazie wiedzy |

## 🔒 Funkcje bezpieczeństwa

### Guardrails (zabezpieczenia)
- **Wykrywanie prompt injection**: Blokuje "ignore instructions", "reveal prompt", próby jailbreak
- **Ochrona przed path traversal**: Blokuje `..`, ścieżki absolutne, wrażliwe pliki
- **Białe listy**: Dostęp tylko do zatwierdzonych hostów i usług
- **Timeouty**: Konfigurowalne limity czasowe dla narzędzi
- **Brak stacktrace'ów**: Komunikaty błędów nie ujawniają wewnętrznych szczegółów

### Testowanie bezpieczeństwa
```bash
# Uruchom testy red-team
pytest tests/test_redteam.py -v
```

## 📊 Uruchamianie testów

```bash
# Wszystkie testy
pytest tests/ -v

# Konkretne pliki testowe
pytest tests/test_tools.py -v      # Walidacja narzędzi
pytest tests/test_redteam.py -v    # Testy bezpieczeństwa
pytest tests/test_rag.py -v        # Jakość RAG

# Z pokryciem kodu
pytest tests/ --cov=app --cov-report=html
```

## 📁 Struktura projektu

```
devops_agent/
├── app/
│   ├── main.py              # Aplikacja FastAPI
│   ├── core/
│   │   ├── agent.py         # Główny agent z pętlą FC
│   │   ├── llm_service.py   # Providery LLM (Gemini, Groq, local)
│   │   └── security.py      # Guardrails
│   ├── rag/
│   │   └── engine.py        # FAISS + embeddingi
│   └── tools/
│       ├── registry.py      # Schematy Pydantic
│       └── implementations.py # Implementacje narzędzi
├── data/runbooks/           # Dokumenty bazy wiedzy
├── mock_fs/                 # Symulowany system plików dla logów
├── tests/                   # Zestaw testów
├── .env.template            # Szablon konfiguracji
└── requirements.txt         # Zależności
```

## 🔧 Opcje konfiguracji

| Zmienna | Domyślnie | Opis |
|---------|-----------|------|
| `LLM_PROVIDER` | `gemini` | Główny LLM (gemini/groq/local) |
| `GOOGLE_API_KEY` | - | Klucz API Gemini |
| `GROQ_API_KEY` | - | Klucz API Groq |
| `TOOL_TIMEOUT_SEC` | `5.0` | Timeout wykonania narzędzi |
| `MAX_RAG_RESULTS` | `5` | Maks. wyników RAG |
| `ALLOWED_HOSTS` | localhost,google.com,... | Hosty dozwolone do pingowania |
| `ALLOWED_SERVICES` | nginx,docker,... | Usługi do zarządzania |
| `USE_RERANKING` | `1` | Włącz reranking cross-encoder |

## 📝 Przykładowe zapytania

```
"Sprawdź logi nginx pod kątem błędów"
"Czy serwer bazy danych działa?"
"API zwraca błędy 502, pomóż mi zdiagnozować problem"
"Ile miejsca na dysku zostało?"
"Pinguj google.com żeby sprawdzić łączność"
"Zrestartuj usługę docker"
```

## 🏆 Punktacja projektu

| Wymaganie | Punkty | Status |
|-----------|--------|--------|
| Rejestr narzędzi + schematy | 15 | ✅ |
| Dispatcher + bezpieczeństwo | 15 | ✅ |
| Function-calling | 15 | ✅ |
| Mini-RAG | 20 | ✅ |
| Guardrails | 9 | ✅ |
| Ewaluacja | 8 | ✅ |
| REST API | 8 | ✅ |
| Observability | 5 | ✅ |
| Jakość kodu | 5 | ✅ |
| Demo/Raport | 10 | ✅ |
| **Bonus: Reranking** | +8 | ✅ |

## 📜 Licencja

MIT
