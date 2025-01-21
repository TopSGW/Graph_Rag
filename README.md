# Graph RAG Agent

A Graph-based Retrieval Augmented Generation (RAG) Agent using LangChain and Neo4j for intelligent document processing and knowledge graph creation.

## Features

- Document processing and text extraction
- Knowledge graph creation using Neo4j
- RAG-based query answering
- Integration with LangChain for NLP tasks
- Efficient document retrieval using graph-based approaches

## Project Structure

```
graph-rag-agent/
├── data/                    # Training and test data
├── src/                     # Source code
│   ├── utils/              # Utility functions
│   │   ├── document_loader.py
│   │   ├── neo4j_helper.py
│   │   └── ollama_helper.py
│   └── main.py             # Main application entry point
├── tests/                  # Test files
├── docs/                   # Documentation
├── config/                 # Configuration files
├── .gitignore             # Git ignore file
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # Project documentation
```

## Prerequisites

- Python 3.9 or higher
- Neo4j Database
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/graph-rag-agent.git
cd graph-rag-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage

1. Start the agent:
```bash
python src/main.py
```

2. Process documents:
```python
from src.utils.document_loader import DocumentLoader
loader = DocumentLoader()
documents = loader.load_documents("path/to/documents")
```

3. Query the knowledge graph:
```python
from src.utils.neo4j_helper import Neo4jHelper
neo4j = Neo4jHelper()
results = neo4j.query("YOUR_CYPHER_QUERY")
```

## Development

### Running Tests
```bash
pytest
```

### Code Style
This project uses:
- Black for code formatting
- isort for import sorting
- pytest for testing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain team for their excellent NLP framework
- Neo4j team for their graph database
- All contributors to this project