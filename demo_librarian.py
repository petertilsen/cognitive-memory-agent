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
    model = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    print(f"✅ Using Bedrock model: {model}")
    
    return True


def demo_basic_research():
    """Demonstrate basic research capabilities."""
    print("\n" + "="*60)
    print("📚 BASIC RESEARCH DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize LibrarianAgent
        print("Initializing LibrarianAgent...")
        librarian = LibrarianAgent()
        print("✅ LibrarianAgent initialized successfully")
        
        # Research queries
        queries = [
            "What is machine learning and how does it work?",
            "Tell me about neural networks and deep learning",
            "How do transformers work in natural language processing?"
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
            "What are the basics of artificial intelligence?",
            "How do machine learning algorithms learn from data?",
            "What makes deep learning different from traditional ML?",
            "How do transformers revolutionize natural language processing?"
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
            print(f"Memory Reuse Rate: {after_report['reuse_analysis']['reuse_rate']:.1%}")
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
    
    # Run demonstrations
    librarian = demo_basic_research()
    if librarian:
        demo_memory_analysis(librarian)
        demo_progressive_research(librarian)
        
        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETE")
        print("="*60)
        print("Key Cognitive Memory Advantages Demonstrated:")
        print("• Persistent memory across queries")
        print("• Progressive knowledge building")
        print("• Memory reuse and consolidation")
        print("• Intelligent buffer management")
        print("• Semantic clustering and organization")
    else:
        print("\n❌ Demo failed. Please check your configuration and try again.")


if __name__ == "__main__":
    main()
