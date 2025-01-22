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

                # Create indexes for better performance
                session.run("""
                    CREATE INDEX document_type IF NOT EXISTS
                    FOR (d:Document) ON (d.file_type)
                """)
                
                session.run("""
                    CREATE INDEX document_created IF NOT EXISTS
                    FOR (d:Document) ON (d.created_at)
                """)

                session.run("""
                    CREATE INDEX document_content IF NOT EXISTS
                    FOR (d:Document) ON (d.text)
                """)

                session.run("""
                    CREATE INDEX document_metadata IF NOT EXISTS
                    FOR (d:Document) ON (d.metadata)
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
        Create relationships between documents using multiple criteria:
        1. Vector similarity (using GDS nodeSimilarity)
        2. Content-based relationships
        3. Temporal relationships
        4. Metadata relationships
        5. File type relationships
        """
        try:
            if not self.check_gds_plugin():
                print("Neo4j Graph Data Science plugin is not available.")
                return False

            with self.driver.session(database=self.database) as session:
                # Drop existing graph if any
                self.safe_drop_graph("doc_graph")

                # Project graph with all necessary properties
                project_query = """
                CALL gds.graph.project(
                    'doc_graph',
                    {
                        Document: {
                            properties: {
                                embedding: {
                                    property: 'embedding',
                                    defaultValue: []
                                },
                                text: {
                                    property: 'text',
                                    defaultValue: ''
                                },
                                file_type: {
                                    property: 'file_type',
                                    defaultValue: ''
                                },
                                created_at: {
                                    property: 'created_at',
                                    defaultValue: datetime()
                                },
                                metadata: {
                                    property: 'metadata',
                                    defaultValue: {}
                                }
                            }
                        }
                    },
                    {
                        SIMILAR: {
                            orientation: 'UNDIRECTED',
                            properties: {
                                score: {
                                    property: 'score',
                                    defaultValue: 0.0
                                }
                            }
                        },
                        SHARES_TYPE: {
                            orientation: 'UNDIRECTED'
                        },
                        TEMPORAL: {
                            orientation: 'UNDIRECTED',
                            properties: {
                                time_diff: {
                                    property: 'time_diff',
                                    defaultValue: 0
                                }
                            }
                        },
                        RELATED_CONTENT: {
                            orientation: 'UNDIRECTED',
                            properties: {
                                relevance: {
                                    property: 'relevance',
                                    defaultValue: 0.0
                                }
                            }
                        }
                    }
                )
                """
                session.run(project_query)

                # Create vector similarity relationships
                session.run("""
                CALL gds.nodeSimilarity.write('doc_graph', {
                    writeRelationshipType: 'SIMILAR',
                    writeProperty: 'score',
                    similarityMetric: 'cosine',
                    topK: 5,
                    similarityCutoff: 0.7
                })
                """)

                # Create content-based relationships
                session.run("""
                MATCH (d1:Document), (d2:Document)
                WHERE id(d1) < id(d2)
                WITH d1, d2,
                     gds.similarity.cosine(
                         split(d1.text, ' '),
                         split(d2.text, ' ')
                     ) AS textSimilarity
                WHERE textSimilarity > 0.3
                CREATE (d1)-[:RELATED_CONTENT {relevance: textSimilarity}]->(d2)
                """)

                # Create temporal relationships
                session.run("""
                MATCH (d1:Document), (d2:Document)
                WHERE id(d1) < id(d2)
                WITH d1, d2,
                     duration.between(
                         datetime(d1.created_at),
                         datetime(d2.created_at)
                     ).days AS daysDiff
                WHERE daysDiff <= 30
                CREATE (d1)-[:TEMPORAL {time_diff: daysDiff}]->(d2)
                """)

                # Create file type relationships
                session.run("""
                MATCH (d1:Document), (d2:Document)
                WHERE d1.file_type = d2.file_type AND id(d1) < id(d2)
                CREATE (d1)-[:SHARES_TYPE]->(d2)
                """)

                # Clean up
                self.safe_drop_graph("doc_graph")

            return True
        except Exception as e:
            print(f"Error creating relationships: {str(e)}")
            self.safe_drop_graph("doc_graph")
            return False

    def similarity_search_with_graph(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid search combining vector similarity with graph traversal.
        Uses both vector search and graph relationships for better context.
        """
        hybrid_query = """
        // First, perform vector similarity search
        WITH $query AS query
        CALL db.index.vector.queryNodes('document_store', $k, query)
        YIELD node, score

        // Get directly similar documents
        OPTIONAL MATCH (node)-[sim:SIMILAR]->(similar:Document)
        
        // Get content-related documents
        OPTIONAL MATCH (node)-[rel:RELATED_CONTENT]->(related:Document)
        WHERE related <> similar
        
        // Get temporal neighbors
        OPTIONAL MATCH (node)-[temp:TEMPORAL]->(temporal:Document)
        WHERE temporal <> similar AND temporal <> related
        
        // Get type-related documents
        OPTIONAL MATCH (node)-[:SHARES_TYPE]->(typeRelated:Document)
        WHERE typeRelated <> similar AND typeRelated <> related AND typeRelated <> temporal
        
        WITH node, score,
             collect(DISTINCT {
                 text: similar.text,
                 similarity: sim.score,
                 type: 'vector_similar',
                 metadata: similar.metadata
             }) AS similar_docs,
             collect(DISTINCT {
                 text: related.text,
                 relevance: rel.relevance,
                 type: 'content_related',
                 metadata: related.metadata
             }) AS related_docs,
             collect(DISTINCT {
                 text: temporal.text,
                 time_diff: temp.time_diff,
                 type: 'temporal',
                 metadata: temporal.metadata
             }) AS temporal_docs,
             collect(DISTINCT {
                 text: typeRelated.text,
                 type: 'same_type',
                 metadata: typeRelated.metadata
             }) AS type_docs
        
        RETURN node.text AS text,
               score,
               {
                   source: node.source,
                   file_type: node.file_type,
                   created_at: node.created_at,
                   metadata: node.metadata,
                   related_documents: similar_docs + related_docs + temporal_docs + type_docs
               } AS metadata
        ORDER BY score DESC
        """
        try:
            # Embed the query text
            query_embedding = self.embeddings.embed_query(query)
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    hybrid_query,
                    query=query_embedding,
                    k=k
                )
                return [record.data() for record in result]
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add new documents and create relationships."""
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
            
            # Insert into vector store
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