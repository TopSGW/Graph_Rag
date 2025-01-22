import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from utils.document_loader import DocumentLoader
from utils.neo4j_helper import Neo4jHelper
from utils.ollama_helper import OllamaHelper
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from datetime import datetime

console = Console()

class RAGSystem:
    def __init__(self):
        # Initialize with the data directory path
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.document_loader = DocumentLoader(directory_path=data_dir)
        self.neo4j_helper = Neo4jHelper()
        self.ollama_helper = OllamaHelper()
        self.vector_store = None
        self.data_dir = data_dir
        
    async def initialize(self):
        """Initialize the RAG system with documents and vector store"""
        try:
            with Progress() as progress:
                # First check if GDS plugin is available
                if not self.neo4j_helper.check_gds_plugin():
                    console.print("[red]Neo4j Graph Data Science plugin is not available.")
                    console.print("[yellow]Please install the GDS plugin in your Neo4j database.")
                    console.print("[yellow]Visit: https://neo4j.com/docs/graph-data-science/current/installation/")
                    return False

                task1 = progress.add_task("[cyan]Loading documents...", total=1)
                # Load documents
                documents = self.document_loader.load_documents()
                if not documents:
                    console.print("[red]No documents found to process")
                    return False
                progress.update(task1, advance=1)

                task2 = progress.add_task("[cyan]Initializing vector store...", total=1)
                # Initialize Neo4j vector store with documents
                self.vector_store = self.neo4j_helper.initialize_vector_store(documents)
                progress.update(task2, advance=1)

                task3 = progress.add_task("[cyan]Creating graph relationships...", total=1)
                # Create graph relationships
                success = self.neo4j_helper.create_graph_relationships()
                if not success:
                    console.print("[red]Failed to create graph relationships")
                    return False
                progress.update(task3, advance=1)

            console.print(Panel.fit(
                "[green]RAG system initialized successfully\n" +
                f"Loaded {len(documents)} document chunks\n" +
                "Vector store and graph relationships created",
                title="Initialization Complete"
            ))
            return True
        except Exception as e:
            console.print(f"[red]Error initializing RAG system: {str(e)}")
            return False

    async def process_query(self, question: str) -> Dict[str, Any]:
        """Process a query and return response with context"""
        try:
            # Get relevant documents with graph context
            results = self.neo4j_helper.similarity_search_with_graph(question)
            
            if not results:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "context": [],
                    "sources": []
                }
            
            # Get response from LLM
            response = await self.ollama_helper.get_rag_response(question, results)
            
            # Format sources
            sources = [
                {
                    "title": result["metadata"].get("source", "").split("/")[-1],
                    "related_docs": [
                        doc["text"][:200] + "..."
                        for doc in result["metadata"].get("related_documents", [])
                    ]
                }
                for result in results
            ]
            
            return {
                "answer": response,
                "context": [r["text"] for r in results],
                "sources": sources
            }
        except Exception as e:
            console.print(f"[red]Error processing query: {str(e)}")
            return {
                "answer": "An error occurred while processing your query.",
                "context": [],
                "sources": []
            }

    def upload_file(self, file_path: str) -> bool:
        """Upload a file to the data directory"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                console.print(f"[red]File not found: {file_path}")
                return False

            # Create destination path
            dest_path = Path(self.data_dir) / source_path.name
            
            # Copy file to data directory
            shutil.copy2(source_path, dest_path)
            
            # Process the new file
            doc = self.document_loader.read_single_document(dest_path.name)
            if doc:
                # Add to vector store
                self.neo4j_helper.add_documents([doc.page_content], [doc.metadata])
                console.print(f"[green]Successfully uploaded and processed: {source_path.name}")
                return True
            else:
                console.print(f"[red]Failed to process file: {source_path.name}")
                return False
        except Exception as e:
            console.print(f"[red]Error uploading file: {str(e)}")
            return False

    def list_files(self, filter_type: str = None) -> None:
        """List all files with optional type filter"""
        try:
            stats = self.document_loader.get_document_stats()
            
            # Create table
            table = Table(title="Document Repository")
            table.add_column("Filename", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Size", style="green")
            table.add_column("Last Modified", style="yellow")
            
            for file_info in stats["files"]:
                if not filter_type or filter_type.lower() in file_info["content_type"].lower():
                    table.add_row(
                        file_info["name"],
                        file_info["content_type"],
                        f"{file_info['size'] / 1024:.2f} KB",
                        datetime.fromisoformat(file_info["last_modified"]).strftime("%Y-%m-%d %H:%M:%S")
                    )
            
            console.print(table)
            console.print(f"\nTotal Files: {stats['total_files']}")
            console.print(f"Total Size: {stats['total_size'] / 1024:.2f} KB")
        except Exception as e:
            console.print(f"[red]Error listing files: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.neo4j_helper.close()
        except Exception as e:
            console.print(f"[red]Error during cleanup: {str(e)}")

async def setup_environment():
    """Setup environment and configuration"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'

    # Create .env file if it doesn't exist
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("""# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=smartAq!1
NEO4J_DATABASE=neo4j

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3.3:70b
OLLAMA_TEMPERATURE=0.1
OLLAMA_MAX_TOKENS=4096

# Document Processing Configuration
DATA_DIR=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=graph_rag.log""")
    
    # Load environment variables
    load_dotenv(dotenv_path=env_path)

def show_help():
    """Show help message with examples"""
    help_text = """
    Commands:
    1. exit - Exit the system
    2. help - Show this help message
    3. upload <file_path> - Upload a new file
    4. list [type] - List all files or filter by type
    
    Example Questions:
    - File Listing:
      * "show me all my tax files"
      * "list documents related to accounting"
    
    - Financial Analysis:
      * "check my net income from 2022"
      * "calculate my business expenses"
    
    - Time-based Queries:
      * "how much tax have I paid over the years"
      * "show my expenses for last year"
    
    - Person-specific Queries:
      * "list my daughter Sophia's tax files"
      * "find documents related to client Sam"
    
    - Email/Communication:
      * "how many emails do I have from client Sam"
      * "show messages about the investment project"
    
    - Image Analysis:
      * "how many horses are in this image"
      * "list my pictures from Paris"
    
    - Document Search:
      * "find files under client name Ashley"
      * "search for documents about investments"
    """
    console.print(Panel.fit(help_text, title="Help"))

async def main():
    try:
        # Setup environment
        await setup_environment()
        
        # Initialize the RAG system
        console.print(Panel.fit(
            "[yellow]Initializing RAG system...\n" +
            "This may take a few minutes",
            title="Setup"
        ))
        
        rag_system = RAGSystem()
        success = await rag_system.initialize()
        
        if not success:
            console.print("[red]Failed to initialize RAG system")
            return

        # Main interaction loop
        console.print(Panel.fit(
            "[green]RAG System Ready!\n" +
            "Type 'exit' to quit\n" +
            "Type 'help' for commands and examples",
            title="System Ready"
        ))
        
        while True:
            try:
                command = input("\nEnter command or question: ").strip()
                
                if not command:
                    continue
                    
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                
                if cmd == 'exit':
                    break
                elif cmd == 'help':
                    show_help()
                elif cmd == 'upload' and len(parts) > 1:
                    rag_system.upload_file(parts[1])
                elif cmd == 'list':
                    filter_type = parts[1] if len(parts) > 1 else None
                    rag_system.list_files(filter_type)
                else:
                    # Process as a question
                    with Progress() as progress:
                        task = progress.add_task("[cyan]Processing query...", total=1)
                        result = await rag_system.process_query(command)
                        progress.update(task, advance=1)
                    
                    # Display results
                    console.print(Panel.fit(
                        result["answer"],
                        title="Answer"
                    ))
                    
                    if result["sources"]:
                        console.print(Panel.fit(
                            "\n".join([
                                f"[cyan]Source: {s['title']}\n" +
                                "[dim]Related documents:[/dim]\n" +
                                "\n".join([f"- {r}" for r in s["related_docs"]])
                                for s in result["sources"]
                            ]),
                            title="Sources"
                        ))
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}")
    
    finally:
        # Cleanup
        if 'rag_system' in locals():
            rag_system.cleanup()
        console.print("[yellow]System shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...")
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}")