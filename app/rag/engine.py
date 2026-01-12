"""
RAG Engine - Mini-RAG z FAISS i sentence-transformers.

Punkty:
- (11 pkt) Embeddingi + indeks FAISS
- (5 pkt) Retriever Top-k + MMR/reranking
- (4 pkt) Pakowanie kontekstu z metadanymi

Bonus:
- (+8 pkt) Reranking z cross-encoder
"""
import os
import re
import glob
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "runbooks")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
USE_RERANKING = os.getenv("USE_RERANKING", "1") == "1"


# ============================================================
# Document Model
# ============================================================

@dataclass
class Document:
    """Dokument z bazy wiedzy."""
    source: str
    chunk_id: int
    text: str
    start_char: int = 0
    end_char: int = 0


# ============================================================
# RAG Engine
# ============================================================

class RAGEngine:
    """
    Mini-RAG Engine z FAISS.
    
    Features:
    - Ładowanie dokumentów (MD, TXT, PDF)
    - Chunking z overlap
    - Embeddingi (sentence-transformers)
    - Indeks FAISS (cosine similarity)
    - Retrieval Top-k
    - Opcjonalny reranking (cross-encoder)
    """
    
    def __init__(
        self,
        data_dir: str = None,
        embedding_model: str = None,
        use_reranking: bool = None,
    ):
        self.data_dir = data_dir or DATA_DIR
        self.embedding_model_name = embedding_model or EMB_MODEL
        self.use_reranking = use_reranking if use_reranking is not None else USE_RERANKING
        
        self.embedder = None
        self.reranker = None
        self.index = None
        self.documents: List[Document] = []
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """Lazy initialization - ładuje modele i buduje indeks."""
        if self._initialized:
            return True
        
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            t0 = time.time()
            self.embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedder loaded in {time.time() - t0:.2f}s")
            
            # Load reranker if enabled
            if self.use_reranking:
                try:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
                    logger.info("Cross-encoder reranker loaded")
                except Exception as e:
                    logger.warning(f"Reranker not available: {e}")
                    self.use_reranking = False
            
            # Load and index documents
            self._load_documents()
            self._build_index()
            
            self._initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install with: pip install sentence-transformers faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False
    
    def _load_documents(self):
        """Ładuje dokumenty z data_dir."""
        self.documents = []
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Find all supported files
        patterns = ["*.md", "*.txt"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.data_dir, "**", pattern), recursive=True))
        
        if not files:
            logger.warning(f"No documents found in {self.data_dir}")
            # Create demo documents
            self._create_demo_docs()
            return self._load_documents()
        
        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                source = os.path.relpath(filepath, self.data_dir)
                chunks = self._chunk_text(content)
                
                for i, (start, end, text) in enumerate(chunks):
                    self.documents.append(Document(
                        source=source,
                        chunk_id=i,
                        text=text.strip(),
                        start_char=start,
                        end_char=end,
                    ))
                    
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} chunks from {len(files)} files")
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None,
    ) -> List[Tuple[int, int, str]]:
        """Dzieli tekst na chunki z overlap."""
        chunk_size = chunk_size or CHUNK_SIZE
        overlap = overlap or CHUNK_OVERLAP
        
        chunks = []
        i = 0
        
        while i < len(text):
            end = min(len(text), i + chunk_size)
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for sep in [". ", ".\n", "\n\n"]:
                    last_sep = text[i:end].rfind(sep)
                    if last_sep > chunk_size // 2:
                        end = i + last_sep + len(sep)
                        break
            
            chunk_text = text[i:end]
            if chunk_text.strip():
                chunks.append((i, end, chunk_text))
            
            if end >= len(text):
                break
            
            i = max(i + 1, end - overlap)
        
        return chunks
    
    def _build_index(self):
        """Buduje indeks FAISS."""
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        import faiss
        
        logger.info("Building FAISS index...")
        t0 = time.time()
        
        texts = [doc.text for doc in self.documents]
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        
        # Create index (Inner Product = cosine similarity for normalized vectors)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}, time={time.time() - t0:.2f}s")
    
    def _create_demo_docs(self):
        """Tworzy przykładowe dokumenty DevOps."""
        demo_docs = {
            "nginx-troubleshooting.md": """# Nginx Troubleshooting Guide

## Error 502 Bad Gateway

### Symptoms
- Users see "502 Bad Gateway" error
- Upstream servers not responding

### Common Causes
1. **Backend server down**: The application server (PHP-FPM, Node.js, etc.) is not running
2. **Socket issues**: Unix socket or TCP connection refused
3. **Timeout**: Backend is too slow to respond
4. **Memory issues**: Backend crashed due to OOM

### Diagnosis Steps
1. Check nginx error logs: `tail -f /var/log/nginx/error.log`
2. Verify backend is running: `systemctl status php-fpm`
3. Check socket/port: `netstat -tlnp | grep 9000`

### Solutions
- Restart backend: `systemctl restart php-fpm`
- Increase proxy timeouts in nginx config
- Check backend memory limits

## Error 504 Gateway Timeout

### Symptoms
- Request takes too long and fails

### Solutions
- Increase `proxy_read_timeout` in nginx config
- Optimize backend queries
- Add caching layer
""",
            "docker-troubleshooting.md": """# Docker Troubleshooting Guide

## Container Exit Code 137 (OOM Killed)

### Symptoms
- Container exits with code 137
- dmesg shows OOM killer invoked

### Causes
- Container memory limit too low
- Memory leak in application
- Peak memory usage exceeded limit

### Solutions
1. Increase memory limit: `docker run -m 2g ...`
2. Add swap: `docker run --memory-swap 4g ...`
3. Find memory leaks in application
4. Use memory profiling tools

## Container Won't Start

### Common Issues
1. **Port conflict**: Port already in use
2. **Volume mount failed**: Path doesn't exist
3. **Image not found**: Pull the image first

### Diagnosis
```bash
docker logs <container_id>
docker inspect <container_id>
docker events
```

## Disk Space Issues

### Symptoms
- "no space left on device" errors
- Container creation fails

### Solutions
```bash
# Remove unused images
docker image prune -a

# Remove stopped containers
docker container prune

# Clean build cache
docker builder prune
```
""",
            "postgresql-troubleshooting.md": """# PostgreSQL Troubleshooting Guide

## Connection Refused

### Symptoms
- "connection refused" error
- Cannot connect to database

### Causes
1. PostgreSQL not running
2. Wrong port/host configuration
3. pg_hba.conf blocking connections
4. Firewall rules

### Solutions
```bash
# Check if running
systemctl status postgresql

# Check listening port
netstat -tlnp | grep 5432

# Check pg_hba.conf
cat /etc/postgresql/*/main/pg_hba.conf
```

## Too Many Connections

### Symptoms
- "FATAL: too many connections" error

### Solutions
1. Increase max_connections in postgresql.conf
2. Use connection pooling (PgBouncer)
3. Close idle connections
4. Check for connection leaks

## Slow Queries

### Diagnosis
1. Enable slow query log
2. Use EXPLAIN ANALYZE
3. Check for missing indexes

### Solutions
- Add appropriate indexes
- Optimize queries
- Increase shared_buffers
- Run VACUUM ANALYZE
""",
            "general-devops.md": """# General DevOps Best Practices

## Service Health Checks

Always implement health check endpoints:
- `/health` - basic liveness
- `/ready` - readiness for traffic

## Log Management

### Log Levels
- ERROR: Application errors requiring attention
- WARNING: Potential issues
- INFO: Normal operations
- DEBUG: Detailed debugging info

### Log Rotation
Configure logrotate to prevent disk filling:
```
/var/log/app/*.log {
    daily
    rotate 7
    compress
    missingok
}
```

## Monitoring Essentials

Key metrics to monitor:
1. CPU usage
2. Memory usage
3. Disk space
4. Network I/O
5. Application response time
6. Error rates

## Incident Response

1. **Detect**: Automated alerts
2. **Respond**: On-call engineer notified
3. **Mitigate**: Reduce impact
4. **Resolve**: Fix root cause
5. **Learn**: Post-mortem
""",
        }
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        for filename, content in demo_docs.items():
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        
        logger.info(f"Created {len(demo_docs)} demo documents in {self.data_dir}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Wyszukuje dokumenty podobne do query.
        
        Args:
            query: Zapytanie
            top_k: Liczba wyników
            rerank: Czy użyć rerankingu (domyślnie z konfiguracji)
        
        Returns:
            Lista (score, document_dict)
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        if not self.documents or self.index is None:
            return []
        
        rerank = rerank if rerank is not None else self.use_reranking
        
        # Get more candidates if reranking
        fetch_k = top_k * 3 if rerank and self.reranker else top_k
        fetch_k = min(fetch_k, len(self.documents))
        
        # Encode query
        query_vec = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        
        # Search
        scores, indices = self.index.search(query_vec, fetch_k)
        
        results = []
        for i in range(fetch_k):
            idx = indices[0][i]
            if idx < 0:
                continue
            
            doc = self.documents[idx]
            results.append((
                float(scores[0][i]),
                {
                    "source": doc.source,
                    "chunk_id": doc.chunk_id,
                    "chunk": doc.text,
                    "start": doc.start_char,
                    "end": doc.end_char,
                }
            ))
        
        # Rerank with cross-encoder
        if rerank and self.reranker and len(results) > top_k:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[float, Dict[str, Any]]],
        top_k: int,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Reranking z cross-encoder."""
        if not self.reranker:
            return candidates[:top_k]
        
        pairs = [[query, c[1]["chunk"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        reranked = [(float(s), c[1]) for s, c in zip(scores, candidates)]
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        return reranked[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Zwraca statystyki indeksu."""
        return {
            "initialized": self._initialized,
            "documents_count": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_model": self.embedding_model_name,
            "reranking_enabled": self.use_reranking and self.reranker is not None,
            "data_dir": self.data_dir,
        }


# ============================================================
# Singleton instance
# ============================================================

_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Zwraca singleton RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        _rag_engine.initialize()
    return _rag_engine


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = RAGEngine()
    engine.initialize()
    
    print("\n=== RAG Stats ===")
    print(engine.get_stats())
    
    print("\n=== Test Queries ===")
    
    queries = [
        "How to fix 502 Bad Gateway in nginx?",
        "Docker container OOM killed what to do?",
        "PostgreSQL too many connections",
        "How to check disk usage?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = engine.retrieve(query, top_k=2)
        for score, doc in results:
            print(f"  [{score:.3f}] {doc['source']}: {doc['chunk'][:100]}...")
