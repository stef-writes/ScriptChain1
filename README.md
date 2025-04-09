# Gaffer

A powerful workflow orchestration engine for AI-powered applications.

## Core Components

### ScriptChain
The ScriptChain is the core orchestration engine that manages the execution of AI workflows. It provides:

- Level-based parallel execution
- Robust context management
- Input validation and formatting
- Error handling and recovery
- Callback integration for monitoring

### Node System
The node system consists of specialized nodes that handle different aspects of AI processing:

- `TextGenerationNode`: Handles text generation with LLMs
  - Async execution
  - Context-aware prompting
  - Input validation
  - Format-specific handling (Text, JSON, Markdown, Code)
  - Token usage tracking

### Context Management
The context management system provides:

- Vector store integration for semantic search and context optimization
- Support for multiple vector store implementations (Pinecone, Pinecone Inference)
- Context versioning and retrieval
- Context window management
- Format-specific context rules
- Input validation and transformation
- Token usage optimization
- Customizable context formatting (Text, JSON, Markdown, Code)

### Workflow Execution
The workflow execution system features:

- Level-based processing
- Parallel execution within levels
- Context propagation
- Error handling and recovery
- Execution monitoring

## Configuration

### LLM Configuration
```python
llm_config = LLMConfig(
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=500,
    max_context_tokens=1000
)
```

### Node Configuration
```python
node_config = NodeConfig(
    id="node1",
    type="llm",
    model="gpt-4",
    prompt="Your prompt with {input}",
    level=0,
    context_rules={
        "input": ContextRule(
            include=True,
            format=ContextFormat.TEXT,
            required=True
        )
    },
    format_specifications={
        "input": {"prefix": "Input: "}
    }
)
```

### Vector Store Configuration
```python
vector_store_config = VectorStoreConfig(
    index_name="your-index",
    environment="your-env",
    dimension=384,
    metric=SimilarityMetric.COSINE,
    pod_type="p1",
    replicas=1,
    use_inference=True,
    inference_model="llama-text-embed-v2",
    api_key="your-api-key",
    host="your-host-url"
)
```

## Testing Framework
The testing framework provides comprehensive coverage of:

- Node initialization and configuration
- Context management and formatting
- Input validation
- Parallel execution
- Error handling
- Callback integration

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run tests:
```bash
python -m pytest tests/
```

## Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 