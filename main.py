"""Main application entry point."""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from src import CognitiveMemoryAgent
from config.settings import get_logger

# Initialize logging
logger = get_logger("main")


def main():
    """Run the cognitive memory agent demo."""
    # Load environment variables
    load_dotenv()
    logger.info("Starting Cognitive Memory Agent application")
    
    # Initialize agent
    try:
        agent = CognitiveMemoryAgent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        return
    
    # Interactive demo
    logger.info("Starting interactive demo session")
    
    interaction_count = 0
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            interaction_count += 1
            logger.debug(f"User interaction #{interaction_count}: {user_input[:100]}...")
            
            if user_input.lower() in ['quit', 'exit']:
                logger.info(f"Session ended by user after {interaction_count} interactions")
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
            logger.debug(f"Agent response length: {len(response)}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            logger.info(f"Session interrupted by user after {interaction_count} interactions")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"Error during interaction #{interaction_count}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
