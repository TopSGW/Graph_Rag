import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
from utils.document_loader import DocumentLoader
from utils.neo4j_helper import Neo4jHelper
from utils.ollama_helper import OllamaHelper
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

console = Console()

class RAGSystem:
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.neo4j_helper = Neo4jHelper()
        self.ollama_helper = OllamaHelper()
        self.vector_store = None
        
    async def initialize(self):
        """Initialize the RAG system with documents and vector store"""
        try:
            with Progress() as progress:
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

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.neo4j_helper.close()
        except Exception as e:
            console.print(f"[red]Error during cleanup: {str(e)}")

async def setup_environment():
    """Setup environment and configuration"""
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j""")
    
    # Load environment variables
    load_dotenv()

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
            "Type 'help' for commands",
            title="System Ready"
        ))
        
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                
                if question.lower() == 'exit':
                    break
                elif question.lower() == 'help':
                    console.print(Panel.fit(
                        "Available commands:\n" +
                        "exit - Exit the system\n" +
                        "help - Show this help message\n" +
                        "Or enter your question about accounting concepts",
                        title="Help"
                    ))
                    continue
                elif not question:
                    continue
                
                with Progress() as progress:
                    task = progress.add_task("[cyan]Processing query...", total=1)
                    result = await rag_system.process_query(question)
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