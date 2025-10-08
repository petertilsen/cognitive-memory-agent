# Cognitive Memory Agent

A production-ready Strands Agents SDK integration with cognitive workspace memory management, implementing active memory systems and ReAct patterns with persistent ChromaDB storage.

## Features

- **🧠 Active Memory Management**: Proactive information storage and retrieval
- **🔄 Persistent Memory**: ChromaDB-based vector storage
- **⚡ ReAct Pattern**: Reason-Act-Observe cycle with memory integration
- **📚 Multi-layered Buffers**: Immediate, working, and episodic memory systems
- **🎯 Cognitive State Tracking**: Task decomposition and confidence scoring

## Prerequisites

- **ChromaDB Server**: Running ChromaDB instance (see [ChromaDB Documentation](https://docs.trychroma.com/))
- **Python 3.8+**

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials and ChromaDB connection details
```

## Configuration

Create a `.env` file with your configuration:

```env
# AWS Bedrock
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-haiku-20240307-v1:0

# ChromaDB Connection
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=cognitive_memory
```

## Usage

### Prerequisites
Ensure ChromaDB server is running and accessible at the configured host/port.

### Command Line Interface
```bash
python main.py
```

### Programmatic Usage
```python
from src import CognitiveMemoryAgent
from src.memory.memory_system import CognitiveMemorySystem

# Initialize memory system (connects to ChromaDB)
memory_system = CognitiveMemorySystem()

# Process tasks with persistent memory
result = memory_system.process_task(
    task="Research machine learning trends",
    documents=["document1.txt", "document2.txt"]
)

# Initialize agent
agent = CognitiveMemoryAgent()
response = agent("I'm working on a machine learning project")
```

## Architecture

```
src/
├── core/
│   └── agent.py          # Main agent implementation
└── memory/               # Complete memory system
    ├── memory_system.py  # Core memory management
    ├── models.py         # Data models
    └── vector_store.py   # ChromaDB integration
```

## Memory System

The cognitive memory system implements:

1. **Immediate Buffer** (8 items): High-frequency access memory
2. **Working Buffer** (64 items): Task-specific working memory  
3. **Episodic Buffer** (256 items): Long-term conversation memory
4. **Vector Store** (ChromaDB): Persistent semantic search

### Memory Operations

- `process_task(task, documents)`: Main entry point for cognitive processing
- `get_metacognitive_status()`: Check cognitive state and memory utilization

## ChromaDB Integration

- **Persistent Storage**: Memory survives application restarts
- **Vector Search**: Semantic similarity search with embeddings
- **Scalable**: Handles large document collections
- **External Dependency**: Requires running ChromaDB server

### ChromaDB Setup Example

For development/testing, you can run ChromaDB with Docker:

```bash
# Example ChromaDB setup (not included in this project)
docker run -p 8000:8000 chromadb/chroma:latest
```

Refer to [ChromaDB Documentation](https://docs.trychroma.com/) for production deployment options.

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

## License

MIT License
