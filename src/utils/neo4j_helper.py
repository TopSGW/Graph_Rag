from typing import List, Dict, Any
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from datetime import datetime

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
        try:
            if documents:
                # Create constraints and indexes if they don't exist
                self._create_constraints_and_indexes()
                
                return Neo4jVector.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    url=self.url,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    index_name="document_store",
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
                    index_name="document_store",
                    node_label="Document",
                    text_node_property="text",
                    embedding_node_property="embedding"
                )
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            return None

    def _create_constraints_and_indexes(self):
        """Create necessary constraints and indexes"""
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints
                session.run("""
                    CREATE CONSTRAINT document_id IF NOT EXISTS
                    FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT file_path IF NOT EXISTS
                    FOR (f:File) REQUIRE f.path IS UNIQUE
                """)

                # Create indexes
                session.run("""
                    CREATE INDEX document_type IF NOT EXISTS
                    FOR (d:Document) ON (d.file_type)
                """)
                
                session.run("""
                    CREATE INDEX document_created IF NOT EXISTS
                    FOR (d:Document) ON (d.created_at)
                """)
        except Exception as e:
            print(f"Error creating constraints and indexes: {str(e)}")
    
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
                    print(f"Successfully dropped graph '{graph_name}'")
            return True
        except Exception as e:
            print(f"Error dropping graph '{graph_name}': {str(e)}")
            return False

    def create_graph_relationships(self):
        """
        Create relationships between documents based on various criteria:
        1. Vector similarity using GDS nodeSimilarity
        2. File type relationships
        3. Temporal relationships
        4. Content-based relationships
        """
        try:
            # Check if the GDS plugin is available
            if not self.check_gds_plugin():
                print("Neo4j Graph Data Science plugin is not available. Please install it first.")
                return False

            with self.driver.session(database=self.database) as session:
                # 1) Safely drop existing in-memory graph
                self.safe_drop_graph("doc_graph")

                # 2) Project the graph with nodes and their properties
                project_query = """
                CALL gds.graph.project(
                    'doc_graph',
                    {
                        Document: {
                            properties: {
                                embedding: {
                                    property: 'embedding'
                                },
                                file_type: {
                                    property: 'file_type'
                                },
                                created_at: {
                                    property: 'created_at'
                                }
                            }
                        }
                    },
                    {
                        SIMILAR: {
                            orientation: 'UNDIRECTED'
                        },
                        SAME_TYPE: {
                            orientation: 'UNDIRECTED'
                        },
                        TEMPORAL: {
                            orientation: 'UNDIRECTED'
                        }
                    }
                )
                """
                session.run(project_query)

                # 3) Create relationships based on vector similarity
                similarity_query = """
                CALL gds.nodeSimilarity.write('doc_graph', {
                    writeRelationshipType: 'SIMILAR',
                    writeProperty: 'score',
                    similarityMetric: 'COSINE',
                    topK: 5,
                    similarityCutoff: 0.7
                })
                """
                session.run(similarity_query)

                # 4) Create relationships based on file type
                file_type_query = """
                MATCH (d1:Document), (d2:Document)
                WHERE d1.file_type = d2.file_type AND id(d1) < id(d2)
                CREATE (d1)-[:SAME_TYPE]->(d2)
                """
                session.run(file_type_query)

                # 5) Create temporal relationships between documents created within the same timeframe
                temporal_query = """
                MATCH (d1:Document), (d2:Document)
                WHERE datetime(d1.created_at) >= datetime(d2.created_at) - duration('P1D')
                  AND datetime(d1.created_at) <= datetime(d2.created_at) + duration('P1D')
                  AND id(d1) < id(d2)
                CREATE (d1)-[:TEMPORAL {timespan: duration.between(
                    datetime(d1.created_at), 
                    datetime(d2.created_at)
                )}]->(d2)
                """
                session.run(temporal_query)

                # 6) Clean up the in-memory graph
                self.safe_drop_graph("doc_graph")

            return True
        except Exception as e:
            print(f"Error creating relationships: {str(e)}")
            self.safe_drop_graph("doc_graph")
            return False

    def similarity_search_with_graph(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Enhanced similarity search that considers:
        1. Vector similarity
        2. File type relationships
        3. Temporal relationships
        4. Content-based relationships
        """
        retrieval_query = """
        WITH $query AS query
        CALL db.index.vector.queryNodes('document_store', $k, query)
        YIELD node, score
        
        // Get directly similar documents
        OPTIONAL MATCH (node)-[r:SIMILAR]->(similar:Document)
        
        // Get documents of the same type
        OPTIONAL MATCH (node)-[:SAME_TYPE]-(sameType:Document)
        WHERE sameType <> similar
        
        // Get temporally related documents
        OPTIONAL MATCH (node)-[t:TEMPORAL]-(temporal:Document)
        WHERE temporal <> similar AND temporal <> sameType
        
        WITH node, score,
             collect(DISTINCT {
                 text: similar.text,
                 similarity: r.score,
                 type: 'similar'
             }) AS similar_docs,
             collect(DISTINCT {
                 text: sameType.text,
                 type: 'same_type'
             }) AS same_type_docs,
             collect(DISTINCT {
                 text: temporal.text,
                 timespan: t.timespan,
                 type: 'temporal'
             }) AS temporal_docs
        
        RETURN node.text AS text,
               score,
               {
                   source: node.source,
                   file_type: node.file_type,
                   created_at: node.created_at,
                   title: node.title,
                   related_documents: similar_docs + same_type_docs + temporal_docs
               } AS metadata
        ORDER BY score DESC
        """
        try:
            # Embed the query text
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
        """
        Add new documents to the vector store and create relationships.
        """
        try:
            # Prepare the docs
            documents = []
            for i, text in enumerate(texts):
                doc = {
                    "page_content": text,
                    "metadata": metadatas[i] if metadatas else {
                        "source": f"doc_{i}",
                        "created_at": datetime.now().isoformat()
                    }
                }
                documents.append(doc)
            
            # Insert into the Neo4j vector store
            self.initialize_vector_store(documents)
            
            # Create relationships
            self.create_graph_relationships()
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()