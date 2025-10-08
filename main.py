"""Main application entry point."""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from cognitive_memory_agent import CognitiveMemoryAgent


def main():
    """Run the cognitive memory agent demo."""
    # Load environment variables
    load_dotenv()
    
    print("🧠 Cognitive Memory Agent - Strands SDK Integration")
    print("=" * 50)
    
    # Initialize agent
    try:
        agent = CognitiveMemoryAgent()
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Interactive demo
    print("\nStarting interactive demo...")
    print("Type 'quit' to exit, 'status' to check memory status")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'status':
                status_response = agent("Please check and show me the current memory status")
                print(f"Agent: {status_response}")
                continue
            
            if not user_input:
                continue
            
            # Process through cognitive agent
            response = agent(user_input)
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
