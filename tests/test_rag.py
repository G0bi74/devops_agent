"""
Testy RAG - retrieval i jakość wyników.

Przypadki testowe:
- Recall dla retrievalu
- Jakość kontekstu
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGEngine:
    """Testy silnika RAG (2 przypadki merytoryczne)."""
    
    @pytest.fixture
    def rag_engine(self):
        """Fixture dla RAG engine."""
        try:
            from app.rag.engine import RAGEngine
            engine = RAGEngine()
            if engine.initialize():
                return engine
        except Exception as e:
            pytest.skip(f"RAG engine not available: {e}")
        return None
    
    def test_rag_retrieves_relevant_docs(self, rag_engine):
        """Test czy RAG zwraca relevantne dokumenty."""
        if rag_engine is None:
            pytest.skip("RAG not initialized")
        
        # Query o nginx 502
        results = rag_engine.retrieve("nginx 502 bad gateway error", top_k=3)
        
        assert len(results) > 0
        
        # Sprawdź czy wyniki zawierają słowa kluczowe
        all_text = " ".join([r[1]["chunk"].lower() for r in results])
        keywords = ["nginx", "502", "gateway", "upstream", "backend"]
        found = sum(1 for kw in keywords if kw in all_text)
        
        assert found >= 2, "RAG should return documents about nginx errors"
    
    def test_rag_docker_oom_query(self, rag_engine):
        """Test query o Docker OOM."""
        if rag_engine is None:
            pytest.skip("RAG not initialized")
        
        results = rag_engine.retrieve("docker container OOM killed memory", top_k=3)
        
        assert len(results) > 0
        
        # Sprawdź relevancję
        all_text = " ".join([r[1]["chunk"].lower() for r in results])
        keywords = ["docker", "oom", "memory", "137", "container"]
        found = sum(1 for kw in keywords if kw in all_text)
        
        assert found >= 2, "RAG should return documents about Docker OOM"
    
    def test_rag_returns_metadata(self, rag_engine):
        """Test czy RAG zwraca metadane."""
        if rag_engine is None:
            pytest.skip("RAG not initialized")
        
        results = rag_engine.retrieve("troubleshooting", top_k=1)
        
        if results:
            score, doc = results[0]
            assert "source" in doc
            assert "chunk" in doc
            assert "chunk_id" in doc
            assert isinstance(score, float)
    
    def test_rag_top_k_respected(self, rag_engine):
        """Test czy top_k jest respektowane."""
        if rag_engine is None:
            pytest.skip("RAG not initialized")
        
        for k in [1, 3, 5]:
            results = rag_engine.retrieve("error", top_k=k)
            assert len(results) <= k
    
    def test_rag_stats(self, rag_engine):
        """Test statystyk RAG."""
        if rag_engine is None:
            pytest.skip("RAG not initialized")
        
        stats = rag_engine.get_stats()
        
        assert "initialized" in stats
        assert "documents_count" in stats
        assert stats["documents_count"] > 0


class TestRAGQuality:
    """Testy jakości RAG."""
    
    @pytest.fixture
    def rag_engine(self):
        try:
            from app.rag.engine import RAGEngine
            engine = RAGEngine()
            if engine.initialize():
                return engine
        except Exception:
            pytest.skip("RAG engine not available")
        return None
    
    def test_postgres_query_quality(self, rag_engine):
        """Test jakości dla zapytań o PostgreSQL."""
        if rag_engine is None:
            pytest.skip("RAG not initialized")
        
        queries = [
            ("postgresql too many connections", ["connection", "max", "pool"]),
            ("database slow queries", ["slow", "query", "index", "explain"]),
        ]
        
        for query, expected_keywords in queries:
            results = rag_engine.retrieve(query, top_k=3)
            if results:
                all_text = " ".join([r[1]["chunk"].lower() for r in results])
                found = sum(1 for kw in expected_keywords if kw in all_text)
                # At least one keyword should be present
                assert found >= 1, f"Query '{query}' should return relevant results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
