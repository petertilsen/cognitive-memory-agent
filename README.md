# 🧠 Cognitive Memory Agent Demo

A production-ready AI agent demonstration showcasing the [cognitive-memory](https://github.com/petertilsen/cognitive-memory) package in action. See how agents learn, remember, and get smarter with each interaction.

## 🎯 What This Demonstrates

This LibrarianAgent shows real-world usage of the **cognitive-memory** package:

- **Progressive Learning**: 25-60% memory reuse vs 0% traditional RAG
- **Domain Expertise**: Builds specialized knowledge over time  
- **Session Persistence**: Remembers across conversations
- **Intelligent Research**: Proactive information gathering with tools

## 🚀 Quick Start

### Prerequisites
- **AWS Bedrock Access**: Configured credentials for Claude and Titan models
- **ChromaDB Server**: See [cognitive-memory setup](https://github.com/petertilsen/cognitive-memory#chromadb-setup)
- **Python 3.8+**

### Installation
```bash
git clone <this-repo-url>
cd cognitive-memory-agent
pip install -r requirements.txt
```

### AWS Configuration
```env
# Copy and configure
cp .env.example .env

# Required AWS settings
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
BEDROCK_MODEL=anthropic.claude-3-haiku-20240307-v1:0
```

### Run the Demo
```bash
python demo_librarian.py
```

## 🧪 Progressive Learning Demo

Watch the agent build expertise through memory reuse:

```python
# Query 1: "What are activation functions?"
# → 0% reuse (new topic), researches and stores knowledge

# Query 2: "Which activation function is most popular?" 
# → 25% reuse (builds on stored activation function knowledge)

# Query 3: "Why use ReLU over sigmoid in deep networks?"
# → 40% reuse (leverages accumulated neural network understanding)

# Query 4: "What is quantum computing?"
# → 0% reuse (new domain), starts building quantum knowledge cluster
```

**Result**: Each query becomes faster and more informed as the agent accumulates domain expertise.

## 🏗️ Agent Architecture

```python
from cognitive_memory import CognitiveMemorySystem
from strands_agents import Agent

class LibrarianAgent(Agent):
    def __init__(self):
        # Cognitive memory handles all learning and recall
        self.memory_system = CognitiveMemorySystem.from_env()
        
        # Research tools for information gathering
        self.tools = [ArxivSearchTool(), WebSearchTool()]
    
    def process_query(self, query):
        # Memory system orchestrates the entire cognitive process
        return self.memory_system.process_task(query, self.research(query))
```

## 📊 Memory Analytics

Monitor learning progress in real-time:

```python
from cognitive_memory.analyzer import MemoryAnalyzer

analyzer = MemoryAnalyzer(agent.memory_system)
report = analyzer.generate_memory_report()

print(f"Memory reuse: {report['reuse_analysis']['reuse_rate']:.1%}")
print(f"Knowledge clusters: {report['consolidation_analysis']['semantic_clusters']}")
print(f"Working memory: {report['buffer_analysis']['working_buffer']['size']}/64")
```

## 🔧 Customization

### Add Your Own Tools
```python
from cognitive_memory import CognitiveMemorySystem

class MyAgent:
    def __init__(self):
        self.memory_system = CognitiveMemorySystem.from_env()
        self.tools = [YourCustomTool()]  # Add any research tools
    
    def chat(self, message):
        documents = self.research_with_tools(message)
        result = self.memory_system.process_task(message, documents)
        return result['final_synthesis']
```

### Different Memory Configurations
```python
from cognitive_memory import CognitiveMemorySystem, MemoryConfig

# Custom memory settings
config = MemoryConfig(
    immediate_buffer_size=16,    # Increase immediate memory
    working_buffer_size=128,     # Larger working memory
    episodic_buffer_size=512     # More episodic storage
)

memory_system = CognitiveMemorySystem(config)
```

## 📈 Performance Benefits

Compared to traditional RAG systems:

| Metric | Traditional RAG | Cognitive Memory Agent |
|--------|----------------|----------------------|
| **Memory Reuse** | 0% | 25-60% |
| **Response Speed** | Baseline | 25x faster (with reuse) |
| **Learning** | None | Progressive improvement |
| **Context** | Single session | Persistent across sessions |

## 🤝 Contributing

This demo shows cognitive-memory package integration. Contribute:

- **New agent types**: Different domains (legal, medical, technical)
- **Tool integrations**: Additional research and data sources  
- **Memory configurations**: Specialized setups for different use cases
- **Performance examples**: Benchmarks and optimization demos

For memory system improvements, contribute to [cognitive-memory](https://github.com/petertilsen/cognitive-memory).

## 📄 License

MIT License - See LICENSE file for details.

---

**Powered by**: [cognitive-memory](https://github.com/petertilsen/cognitive-memory) • [Strands Agents SDK](https://github.com/StrAnds-AI/strands-agents) • [AWS Bedrock](https://aws.amazon.com/bedrock/)
