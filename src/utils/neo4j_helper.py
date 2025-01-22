from typing import List, Dict, Any
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=env_path)

class Neo4jHelper:
    def __init__(self):
        self.url = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.3:70b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
    
    def initialize_vector_store(self, documents: List[Dict[str, Any]] = None) -> Neo4jVector:
        """Initialize Neo4j Vector store with documents if provided."""
        if documents:
            return Neo4jVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=self.url,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name="accounting_docs",
                node_label="Document",
                text_node_property="text",
                embedding_node_property="embedding",
                embedding_dimension=4096  # dimension for llama3.3:70b embeddings
            )
        else:
            return Neo4jVector.from_existing_index(
                embedding=self.embeddings,
                url=self.url,
                username=self.username,
                password=self.password,
                database=self.database,
                index_name="accounting_docs",
                node_label="Document",
                text_node_property="text",
                embedding_node_property="embedding"
            )
    
    def check_gds_plugin(self) -> bool:
        """Check if the Graph Data Science plugin is installed and available."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN gds.version() AS version")
                version = result.single()["version"]
                print(f"GDS version: {version}")
                return True
        except Exception as e:
            print(f"Error checking GDS plugin: {str(e)}")
            return False
    
    def check_graph_exists(self, graph_name: str) -> bool:
        """Check if a named graph exists in the GDS catalog."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    "CALL gds.graph.exists($name) YIELD exists",
                    name=graph_name
                )
                return result.single()["exists"]
        except Exception as e:
            print(f"Error checking graph existence: {str(e)}")
            return False

    def safe_drop_graph(self, graph_name: str) -> bool:
        """Safely drop a graph if it exists."""
        try:
            if self.check_graph_exists(graph_name):
                with self.driver.session(database=self.database) as session:
                    session.run("CALL gds.graph.drop($name)", name=graph_name)
                    print(f"Successfully dropped graph {graph_name}")
            return True
        except Exception as e:
            print(f"Error dropping graph {graph_name}: {str(e)}")
            return False
    
    def create_graph_relationships(self):
        """Create relationships between similar documents using cosine similarity."""
        try:
            # First check if GDS plugin is available
            if not self.check_gds_plugin():
                print("Neo4j Graph Data Science plugin is not available. Please install it first.")
                return False

            with self.driver.session(database=self.database) as session:
                # Safely handle existing graph
                self.safe_drop_graph("doc_graph")

                # Step 1: Create a graph projection using Cypher projection
                project_query = """
                MATCH (source:Document)
                WITH collect(source) AS nodes
                RETURN gds.graph.project(
                    'doc_graph',
                    nodes,
                    [],
                    {
                        nodeProperties: ['embedding']
                    }
                ) AS graphInfo
                """
                try:
                    result = session.run(project_query)
                    stats = result.single()["graphInfo"]
                    print(f"Graph projected with {stats['nodeCount']} nodes")
                except Exception as e:
                    print(f"Error projecting graph: {str(e)}")
                    return False

                # Step 2: Run node similarity (cosine)
                similarity_query = """
                CALL gds.nodeSimilarity.write('doc_graph', {
                    writeRelationshipType: 'SIMILAR',
                    writeProperty: 'score',
                    similarityMetric: 'COSINE',
                    topK: 5,
                    similarityCutoff: 0.7,
                    concurrency: 4,
                    writeMode: 'WRITE'
                })
                YIELD 
                    nodesCompared,
                    relationshipsWritten,
                    similarityDistribution,
                    computeMillis
                """
                try:
                    result = session.run(similarity_query)
                    stats = result.single()
                    print(f"Nodes compared: {stats['nodesCompared']}")
                    print(f"Relationships written: {stats['relationshipsWritten']}")
                    print(f"Computation time: {stats['computeMillis']}ms")
                    print(f"Similarity distribution: {stats['similarityDistribution']}")
                except Exception as e:
                    print(f"Error computing node similarity: {str(e)}")
                    self.safe_drop_graph("doc_graph")
                    return False

                # Step 3: Clean up (optional, if you do not want to keep the in-memory graph)
                self.safe_drop_graph("doc_graph")

            return True
        except Exception as e:
            print(f"Error creating relationships: {str(e)}")
            self.safe_drop_graph("doc_graph")
            return False

    def similarity_search_with_graph(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform similarity search with graph-enhanced results.
        Returns documents along with their related documents.
        """
        retrieval_query = """
        WITH $query AS query
        CALL db.index.vector.queryNodes('accounting_docs', $k, query)
        YIELD node, score
        OPTIONAL MATCH (node)-[r:SIMILAR]->(related:Document)
        WITH node, score, collect({text: related.text, similarity: r.score}) AS related_docs
        RETURN node.text AS text, 
               score,
               {
                 source: node.source,
                 related_documents: related_docs
               } AS metadata
        ORDER BY score DESC
        """
        
        try:
            vector_store = self.initialize_vector_store()
            query_embedding = self.embeddings.embed_query(query)
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    retrieval_query,
                    query=query_embedding,
                    k=k
                )
                return [record.data() for record in result]
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store and create graph relationships."""
        try:
            # Convert texts and metadata to document format
            documents = []
            for i, text in enumerate(texts):
                doc = {
                    "page_content": text,
                    "metadata": metadatas[i] if metadatas else {"source": f"doc_{i}"}
                }
                documents.append(doc)
            
            # Initialize vector store with documents
            self.initialize_vector_store(documents)
            
            # Create graph relationships
            self.create_graph_relationships()
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
