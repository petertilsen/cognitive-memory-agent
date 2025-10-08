# Cognitive Memory Agent

A production-ready Strands Agents SDK integration with cognitive workspace memory management, implementing active memory systems and ReAct patterns.

## Features

- **🧠 Active Memory Management**: Proactive information storage and retrieval
- **🔄 Persistent State**: Memory persists across conversations  
- **⚡ ReAct Pattern**: Reason-Act-Observe cycle with memory integration
- **📚 Multi-layered Buffers**: Immediate, working, and episodic memory systems
- **🎯 Cognitive State Tracking**: Task decomposition and confidence scoring

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CognitiveMemoryAgent

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
cp .env.example .env
# Edit .env with your AWS credentials
```

### Configuration

Create a `.env` file with your AWS Bedrock configuration:

```env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-haiku-20240307-v1:0
```

### Usage

#### Command Line Interface
```bash
python main.py
```

#### Programmatic Usage
```python
from cognitive_memory_agent import CognitiveMemoryAgent

# Initialize agent
agent = CognitiveMemoryAgent()

# Use the agent
response = agent("I'm working on a machine learning project")
print(response)

# Check memory status
status = agent("What do you remember about my projects?")
print(status)
```

## Architecture

```
src/
├── cognitive_memory_agent/
│   ├── core/
│   │   ├── agent.py          # Main agent implementation
│   │   └── memory_system.py  # Core memory management
│   ├── models/
│   │   └── memory.py         # Data models
│   ├── tools/
│   │   └── memory_tools.py   # Strands tools for memory
│   └── utils/
├── config/
│   └── settings.py           # Configuration management
├── tests/
└── docs/
```

## Memory System

The cognitive memory system implements three layers:

1. **Immediate Buffer** (8 items): High-frequency access memory
2. **Working Buffer** (64 items): Task-specific working memory  
3. **Episodic Buffer** (256 items): Long-term conversation memory

### Memory Operations

- `add_to_memory(content, context)`: Store information with context
- `retrieve_from_memory(query)`: Retrieve relevant memories
- `consolidate_memory()`: Apply forgetting curves and organize
- `get_memory_status()`: Check current memory state

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Type Checking
```bash
mypy src/
```

## Next Steps

- [ ] Step 2: Enhanced vector embeddings integration
- [ ] Step 3: Advanced ReAct pattern implementation  
- [ ] Step 4: Multi-agent coordination
- [ ] Step 5: Production deployment configuration

## License

MIT License
