"""Test runner for all unit tests."""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Discover tests recursively
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

def run_memory_tests():
    """Run only memory system tests."""
    loader = unittest.TestLoader()
    memory_dir = os.path.join(os.path.dirname(__file__), 'memory')
    suite = loader.discover(memory_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_agent_tests():
    """Run only agent system tests."""
    loader = unittest.TestLoader()
    agent_dir = os.path.join(os.path.dirname(__file__), 'agent')
    suite = loader.discover(agent_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run a specific test module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'memory':
            success = run_memory_tests()
        elif command == 'agent':
            success = run_agent_tests()
        else:
            # Run specific test module
            success = run_specific_test(command)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
