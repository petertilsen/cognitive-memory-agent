#!/usr/bin/env python3
"""
LibrarianAgent Demo Script

Prerequisites:
1. ChromaDB server running (default: localhost:8000)
2. AWS credentials configured for Bedrock
3. Environment variables set in .env file

Usage:
    python demo_librarian.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.librarian_agent import LibrarianAgent
from src.memory.analyzer import MemoryAnalyzer


def flush_chromadb(host: str = "localhost", port: int = 8000):
    """Flush ChromaDB collections for fresh start."""
    try:
        import chromadb

        # Connect to ChromaDB
        client = chromadb.HttpClient(host=host, port=port)
        
        # Get collection name from environment
        collection_name = os.getenv("CHROMA_COLLECTION", "cognitive_memory")
        
        # Try to delete the specific collection instead of full reset
        try:
            client.delete_collection(name=collection_name)
            print(f"✅ Deleted collection '{collection_name}' - starting fresh")
        except Exception:
            # Collection might not exist, which is fine
            print(f"✅ Collection '{collection_name}' cleared - starting fresh")
        
    except Exception as e:
        print(f"⚠️  ChromaDB flush failed: {e}")
        print("   Continuing with existing data...")


def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check ChromaDB connection
    try:
        import requests
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = os.getenv("CHROMA_PORT", "8000")
        
        # Try different endpoints (ChromaDB versions use different APIs)
        endpoints = [
            f"http://{chroma_host}:{chroma_port}/api/v1/heartbeat",
            f"http://{chroma_host}:{chroma_port}/heartbeat",
            f"http://{chroma_host}:{chroma_port}/"
        ]
        
        connected = False
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code in [200, 404, 410] or "chroma" in response.text.lower():
                    connected = True
                    break
            except:
                continue
        
        if connected:
            print(f"✅ ChromaDB server accessible at {chroma_host}:{chroma_port}")

            # flush_chromadb(chroma_host, chroma_port)  # Commented out - causes hanging
        else:
            print(f"❌ ChromaDB server not responding at {chroma_host}:{chroma_port}")
            return False
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        print("   Please start ChromaDB server or check CHROMA_HOST/CHROMA_PORT settings")
        return False
    
    # Check AWS credentials
    if not (os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")):
        print("❌ AWS credentials not found")
        print("   Please set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or configure AWS_PROFILE")
        return False
    else:
        print("✅ AWS credentials found")
    
    # Check Bedrock model
    model = os.getenv("MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    print(f"✅ Using Bedrock model: {model}")
    
    return True


def demo_basic_research(librarian):
    """Demonstrate basic research capabilities."""
    print("\n" + "="*60)
    print("📚 BASIC RESEARCH DEMONSTRATION")
    print("="*60)
    
    try:
        # Research queries
        queries = [
            "What is machine learning and how does it work?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Research Query {i} ---")
            print(f"Query: {query}")
            print("Processing...")
            
            response = librarian.research(query)
            print(f"Response: {response[:200]}...")
            
            # Show memory status
            status = librarian.get_memory_status()
            print(f"Memory Status: {status['memory_utilization']}")
        
        return librarian
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


def demo_memory_analysis(librarian):
    """Demonstrate memory analysis capabilities."""
    print("\n" + "="*60)
    print("🧠 MEMORY ANALYSIS DEMONSTRATION")
    print("="*60)
    
    try:
        # Create memory analyzer
        analyzer = MemoryAnalyzer(librarian.memory_system)
        
        # Generate comprehensive report
        print("Generating memory analysis report...")
        report = analyzer.generate_memory_report()
        
        # Display key metrics
        print(f"\n📊 Memory System Metrics:")
        print(f"   Memory Reuse Rate: {report['reuse_analysis']['reuse_rate']:.1%}")
        print(f"   Working Memory Size: {report['buffer_analysis']['working_buffer']['size']}")
        print(f"   Episodic Memory Size: {report['buffer_analysis']['episodic_buffer']['size']}")
        print(f"   Vector Store Size: {report['buffer_analysis']['vector_store']['size']}")
        
        # Show buffer utilization
        print(f"\n🔄 Buffer Utilization:")
        for buffer_name in ['immediate_buffer', 'working_buffer', 'episodic_buffer']:
            buffer_info = report['buffer_analysis'][buffer_name]
            print(f"   {buffer_name.title()}: {buffer_info['utilization']:.1%} ({buffer_info['size']}/{buffer_info['capacity']})")
        
        # Show memory reuse items
        reuse_items = report['reuse_analysis']['high_access_items']
        if reuse_items:
            print(f"\n♻️  High Reuse Items ({len(reuse_items)}):")
            for item in reuse_items[:3]:  # Show top 3
                print(f"   - {item['content']} (accessed {item['access_count']} times)")
        
        # Show consolidation patterns
        consolidation = report['consolidation_analysis']
        print(f"\n🔄 Consolidation Patterns:")
        print(f"   Semantic Clusters: {consolidation['semantic_clusters']}")
        print(f"   Promotion Candidates: {len(consolidation['promotion_candidates'])}")
        print(f"   Items to be Filtered: {len(consolidation['attention_filtered'])}")
        
        return report
        
    except Exception as e:
        print(f"❌ Memory analysis failed: {e}")
        return None


def demo_progressive_research(librarian):
    """Demonstrate progressive research with memory building."""
    print("\n" + "="*60)
    print("🔬 PROGRESSIVE RESEARCH DEMONSTRATION")
    print("="*60)
    
    try:
        # Progressive queries that build on each other
        progressive_queries = [
            "What are activation functions?",
            "which is the most used activation function?",
            "why would I use relu over sigmoid?",
            "What is quantum computing and how does it work?"
        ]
        
        analyzer = MemoryAnalyzer(librarian.memory_system)
        reports = []
        
        for i, query in enumerate(progressive_queries, 1):
            print(f"\n--- Progressive Query {i} ---")
            print(f"Query: {query}")

            # Generate report before processing
            if i > 1:
                before_report = analyzer.generate_memory_report()

            # Process query
            response = librarian.research(query)
            print(f"Response: {response[:150]}...")
            
            # Generate report after processing
            after_report = analyzer.generate_memory_report()
            reports.append(after_report)
            
            # Show memory evolution
            print(f"Overall Reuse Rate: {after_report['reuse_analysis']['reuse_rate']:.1%}")
            print(f"  - Task-level reuse: {after_report['reuse_analysis'].get('task_reuse_operations', 0)} ops")
            print(f"  - Subtask-level reuse: {after_report['reuse_analysis'].get('memory_reuse_operations', 0)} ops")
            print(f"  - New info processing: {after_report['reuse_analysis'].get('new_info_operations', 0)} ops")
            print(f"Working Memory: {after_report['buffer_analysis']['working_buffer']['size']} items")
            
            # Compare with previous state
            if i > 1:
                comparison = analyzer.compare_memory_states(before_report, after_report)
                working_change = comparison['buffer_changes']['working_buffer']['change']
                print(f"Working Memory Change: {working_change:+d} items")
        
        # Show overall progression
        print(f"\n📈 Research Progression Summary:")
        for i, report in enumerate(reports, 1):
            reuse_rate = report['reuse_analysis']['reuse_rate']
            working_size = report['buffer_analysis']['working_buffer']['size']
            print(f"   Query {i}: {reuse_rate:.1%} reuse, {working_size} working items")
        
    except Exception as e:
        print(f"❌ Progressive research demo failed: {e}")


def main():
    """Main demo function."""
    print("🤖 LibrarianAgent Cognitive Memory Demonstration")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return

    # Initialize LibrarianAgent
    print("Initializing LibrarianAgent...")
    librarian = LibrarianAgent()
    print("✅ LibrarianAgent initialized successfully")

    #demo_basic_research(librarian)
    demo_progressive_research(librarian)


if __name__ == "__main__":
    main()
