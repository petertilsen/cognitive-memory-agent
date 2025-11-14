#!/usr/bin/env python3
"""LibrarianAgent Demo - Cognitive Memory System"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.librarian_agent import LibrarianAgent
from cognitive_memory import MemoryAnalyzer


def check_prerequisites():
    """Check ChromaDB and AWS setup."""
    try:
        import requests
        host = os.getenv("CHROMA_HOST", "localhost")
        port = os.getenv("CHROMA_PORT", "8000")
        
        requests.get(f"http://{host}:{port}/api/v1/heartbeat", timeout=3)
        print(f"✅ ChromaDB accessible at {host}:{port}")
        
        # Clear collection
        import chromadb
        client = chromadb.HttpClient(host=host, port=port)
        try:
            #client.delete_collection(os.getenv("CHROMA_COLLECTION", "cognitive_memory"))
            print("✅ Collection cleared")
        except:
            pass
            
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False
    
    if not (os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")):
        print("❌ AWS credentials missing")
        return False
    
    print("✅ AWS credentials found")
    return True


def demo_progressive_research(librarian, analyzer):
    """Progressive research demonstration."""
    queries = [
        "When was the french revolution?",
        "Who are the hugonotes?",
        "When did Napoleon Bonaparte die?"
    ]
    
    print("\n🔬 Progressive Research Demo")
    print("=" * 40)
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        before = analyzer.generate_memory_report() if i > 1 else None
        if response := librarian.research(query):
            print(f"Response: {response[:100]}...")
        after = analyzer.generate_memory_report()

        reuse_stats = analyzer.get_reuse_stats()
        print(f"\n--- Analyzer Result {i}: {query} ---")
        print(f"Reuse Rate: {reuse_stats['reuse_rate']:.1%}")
        print(f"Working Memory: {after['buffer_analysis']['working_buffer']['size']} items")
        
        if before:
            change = after['buffer_analysis']['working_buffer']['size'] - before['buffer_analysis']['working_buffer']['size']
            print(f"Memory Change: {change:+d} items")


def main():
    """Main demo."""
    print("🤖 LibrarianAgent Demo")
    print("=" * 30)
    
    if not check_prerequisites():
        return
    
    print("Initializing...")
    librarian = LibrarianAgent()
    analyzer = MemoryAnalyzer(librarian.memory_system)
    print("✅ Ready")
    
    demo_progressive_research(librarian, analyzer)


if __name__ == "__main__":
    main()
