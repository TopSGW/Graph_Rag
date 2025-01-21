from typing import Dict, Any, Optional
from pathlib import Path
import os
from dotenv import load_dotenv

class Config:
    """Configuration management for the Graph RAG Agent."""
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Neo4j Configuration
        self.neo4j_config = {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', ''),
            'database': os.getenv('NEO4J_DATABASE', 'neo4j'),
        }
        
        # Ollama Configuration
        self.ollama_config = {
            'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            'default_model': os.getenv('OLLAMA_DEFAULT_MODEL', 'llama3.3:70b'),
            'temperature': float(os.getenv('OLLAMA_TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('OLLAMA_MAX_TOKENS', '4096')),
        }
        
        # Document Processing Configuration
        self.document_config = {
            'data_dir': os.getenv('DATA_DIR', str(Path(__file__).parent.parent / 'data')),
            'supported_extensions': ['.txt', '.pdf', '.docx', '.md'],
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
        }
        
        # Logging Configuration
        self.logging_config = {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': os.getenv('LOG_FILE', 'graph_rag.log'),
        }
        
        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration settings."""
        # Validate Neo4j configuration
        if not self.neo4j_config['password']:
            raise ValueError("Neo4j password must be set in environment variables")
        
        # Validate data directory
        data_dir = Path(self.document_config['data_dir'])
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate numeric values
        if not (0 <= self.ollama_config['temperature'] <= 1):
            raise ValueError("Temperature must be between 0 and 1")
        
        if self.ollama_config['max_tokens'] <= 0:
            raise ValueError("Max tokens must be positive")
        
        if self.document_config['chunk_size'] <= 0:
            raise ValueError("Chunk size must be positive")
        
        if self.document_config['chunk_overlap'] < 0:
            raise ValueError("Chunk overlap cannot be negative")
        
        if self.document_config['chunk_overlap'] >= self.document_config['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")

    def get_neo4j_config(self) -> Dict[str, str]:
        """Get Neo4j configuration."""
        return self.neo4j_config

    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration."""
        return self.ollama_config

    def get_document_config(self) -> Dict[str, Any]:
        """Get document processing configuration."""
        return self.document_config

    def get_logging_config(self) -> Dict[str, str]:
        """Get logging configuration."""
        return self.logging_config

    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value.
        
        Args:
            section: Configuration section (neo4j_config, ollama_config, etc.)
            key: Configuration key to update
            value: New value
        """
        if not hasattr(self, section):
            raise ValueError(f"Invalid configuration section: {section}")
        
        config_section = getattr(self, section)
        if key not in config_section:
            raise ValueError(f"Invalid configuration key: {key}")
        
        config_section[key] = value
        self._validate_config()

    @staticmethod
    def get_instance() -> 'Config':
        """Get or create singleton instance of Config."""
        if not hasattr(Config, '_instance'):
            Config._instance = Config()
        return Config._instance

# Create a default instance
config = Config.get_instance()