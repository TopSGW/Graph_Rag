from typing import List, Dict, Any, Optional, Union
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import uuid
import json
import hashlib

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=env_path)

class Neo4jHelper:
    def __init__(self):
        self.url = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.index_name = "accounting_docs"
        self.vector_cache_path = "vector_cache.json"
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.3:70b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
        
        # Initialize vector store and cache
        self._init_vector_store()
        
    def check_gds_plugin(self) -> bool:
        """Check if the Graph Data Science plugin is installed and available."""
        try:
            with self.driver.session(database=self.database) as session:
                # Try to call a GDS function to verify plugin is installed and working
                result = session.run("CALL gds.version() YIELD version")
                version = result.single()["version"]
                print(f"Neo4j Graph Data Science plugin version: {version}")
                return True
        except Exception as e:
            if "not found" in str(e).lower():
                print("Neo4j Graph Data Science plugin is not installed.")
                print("Please install it from: https://neo4j.com/docs/graph-data-science/current/installation/")
            else:
                print(f"Error checking GDS plugin: {str(e)}")
            return False

    def _init_vector_store(self):
        """Initialize vector store and handle initial vectorization"""
        try:
            # Check if vector index exists
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    SHOW VECTOR INDEXES
                    YIELD name
                    WHERE name = $index_name
                    RETURN count(*) as count
                """, index_name=self.index_name)
                
                if result.single()["count"] == 0:
                    # Create vector index
                    session.run("""
                        CALL db.index.vector.createNodeIndex(
                            $index_name,
                            'Document',
                            'embedding',
                            8192,
                            'cosine'
                        )
                    """, index_name=self.index_name)
                    print(f"Created vector index: {self.index_name}")
                    
                    # Check for existing documents without embeddings
                    self._vectorize_existing_documents()
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")

    def _vectorize_existing_documents(self):
        """Vectorize existing documents that don't have embeddings"""
        try:
            with self.driver.session(database=self.database) as session:
                # Get documents without embeddings
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.embedding IS NULL
                    RETURN d.id as id, d.text as text
                """)
                
                documents = [(record["id"], record["text"]) for record in result]
                
                if documents:
                    print(f"Vectorizing {len(documents)} documents...")
                    for doc_id, text in documents:
                        embedding = self.embeddings.embed_query(text)
                        session.run("""
                            MATCH (d:Document {id: $id})
                            SET d.embedding = $embedding
                        """, id=doc_id, embedding=embedding)
                    print("Vectorization complete")
        except Exception as e:
            print(f"Error vectorizing documents: {str(e)}")

    def similarity_search_with_graph(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity with graph traversal.
        
        This implements a hybrid search approach:
        1. First finds similar documents using vector similarity
        2. Then expands the context using graph relationships:
           - Direct vector similarity relationships
           - Content-based relationships
           - Temporal relationships
           - Type-based relationships
        3. Combines and ranks the results
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            with self.driver.session(database=self.database) as session:
                # Execute hybrid search query
                hybrid_query = """
                // 1. Initial Vector Search
                CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                YIELD node, score
                WITH node, score
                
                // 2. Graph Traversal for Context Expansion
                
                // 2.1 Get directly similar documents through vector similarity
                OPTIONAL MATCH (node)-[sim:SIMILAR]->(similar:Document)
                
                // 2.2 Get content-related documents
                OPTIONAL MATCH (node)-[rel:RELATED_CONTENT]->(related:Document)
                WHERE related <> similar
                
                // 2.3 Get temporal neighbors (documents from similar time periods)
                OPTIONAL MATCH (node)-[temp:TEMPORAL]->(temporal:Document)
                WHERE temporal <> similar 
                  AND temporal <> related
                
                // 2.4 Get type-related documents (same document type)
                OPTIONAL MATCH (node)-[:SHARES_TYPE]->(typeRelated:Document)
                WHERE typeRelated <> similar 
                  AND typeRelated <> related 
                  AND typeRelated <> temporal
                
                // 3. Collect and Structure Results
                WITH node, score,
                     // Similar documents by vector similarity
                     collect(DISTINCT {
                         text: similar.text,
                         similarity: sim.score,
                         type: 'vector_similar',
                         metadata: similar.metadata
                     }) AS similar_docs,
                     
                     // Related documents by content
                     collect(DISTINCT {
                         text: related.text,
                         relevance: rel.relevance,
                         type: 'content_related',
                         metadata: related.metadata
                     }) AS related_docs,
                     
                     // Temporal neighbors
                     collect(DISTINCT {
                         text: temporal.text,
                         time_diff: temp.time_diff,
                         type: 'temporal',
                         metadata: temporal.metadata
                     }) AS temporal_docs,
                     
                     // Type-related documents
                     collect(DISTINCT {
                         text: typeRelated.text,
                         type: 'same_type',
                         metadata: typeRelated.metadata
                     }) AS type_docs
                
                // 4. Return Final Results
                RETURN 
                    node.text AS text,
                    score AS vector_score,
                    {
                        source: node.metadata.source,
                        file_type: node.metadata.file_type,
                        created_at: node.metadata.created_at,
                        metadata: node.metadata,
                        // Combine all related documents with their relationship types
                        related_documents: similar_docs + related_docs + temporal_docs + type_docs
                    } AS metadata
                ORDER BY vector_score DESC
                """
                
                # Execute the query with parameters
                result = session.run(
                    hybrid_query,
                    index_name=self.index_name,
                    embedding=query_embedding,
                    k=k
                )
                
                # Process and return results
                documents = []
                for record in result:
                    doc = {
                        "text": record["text"],
                        "score": record["vector_score"],
                        "metadata": record["metadata"]
                    }
                    
                    # Add relationship information
                    if record["metadata"]["related_documents"]:
                        doc["relationships"] = {
                            "similar": [d for d in record["metadata"]["related_documents"] if d["type"] == "vector_similar"],
                            "content": [d for d in record["metadata"]["related_documents"] if d["type"] == "content_related"],
                            "temporal": [d for d in record["metadata"]["related_documents"] if d["type"] == "temporal"],
                            "type": [d for d in record["metadata"]["related_documents"] if d["type"] == "same_type"]
                        }
                    
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []

    def create_graph_relationships(self):
        """Create relationships between documents using multiple criteria"""
        try:
            if not self.check_gds_plugin():
                print("Neo4j Graph Data Science plugin is not available.")
                return False

            with self.driver.session(database=self.database) as session:
                # First create the relationships in the database
                # 1. Create vector similarity relationships using elementId()
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE elementId(d1) < elementId(d2)
                WITH d1, d2, gds.similarity.cosine(d1.embedding, d2.embedding) AS similarity
                WHERE similarity > 0.7
                CREATE (d1)-[:SIMILAR {score: similarity}]->(d2)
                """)

                # 2. Create content-based relationships using embeddings
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE elementId(d1) < elementId(d2)
                WITH d1, d2,
                    gds.similarity.cosine(d1.embedding, d2.embedding) AS contentSimilarity
                WHERE contentSimilarity > 0.3
                CREATE (d1)-[:RELATED_CONTENT {relevance: contentSimilarity}]->(d2)
                """)

                # 3. Create temporal relationships with proper null handling
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE elementId(d1) < elementId(d2)
                AND d1.metadata.created_at IS NOT NULL 
                AND d2.metadata.created_at IS NOT NULL
                WITH d1, d2,
                    CASE
                        WHEN d1.metadata.created_at IS NOT NULL 
                            AND d2.metadata.created_at IS NOT NULL
                        THEN abs(toInteger(d1.metadata.created_at) - toInteger(d2.metadata.created_at)) / 86400
                        ELSE 999999
                    END AS daysDiff
                WHERE daysDiff <= 30
                CREATE (d1)-[:TEMPORAL {time_diff: daysDiff}]->(d2)
                """)

                # 4. Create file type relationships
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE d1.file_type = d2.file_type AND elementId(d1) < elementId(d2)
                CREATE (d1)-[:SHARES_TYPE]->(d2)
                """)

                # Now project the graph with the created relationships
                project_query = """
                CALL gds.graph.project(
                    'doc_graph',
                    'Document',
                    {
                        SIMILAR: {
                            orientation: 'UNDIRECTED',
                            properties: ['score']
                        },
                        RELATED_CONTENT: {
                            orientation: 'UNDIRECTED',
                            properties: ['relevance']
                        },
                        TEMPORAL: {
                            orientation: 'UNDIRECTED',
                            properties: ['time_diff']
                        },
                        SHARES_TYPE: {
                            orientation: 'UNDIRECTED'
                        }
                    },
                    {
                        nodeProperties: ['embedding', 'text', 'file_type', 'metadata']
                    }
                )
                """
                session.run(project_query)

                # Clean up the projected graph
                self.safe_drop_graph("doc_graph")

            return True
        except Exception as e:
            print(f"Error creating relationships: {str(e)}")
            self.safe_drop_graph("doc_graph")
            return False

    # CRUD Operations
    def create_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Create a single document with vector embedding"""
        try:
            doc_id = str(uuid.uuid4())
            embedding = self.embeddings.embed_query(text)
            
            if metadata is None:
                metadata = {}
            metadata.update({
                "id": doc_id,
                "created_at": str(int(datetime.now().timestamp())),
                "updated_at": str(int(datetime.now().timestamp()))
            })
            
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    CREATE (d:Document {
                        id: $id,
                        text: $text,
                        embedding: $embedding,
                        metadata: $metadata
                    })
                    RETURN d.id as id
                """, {
                    "id": doc_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata
                })
                return result.single()["id"]
        except Exception as e:
            print(f"Error creating document: {str(e)}")
            return None

    def read_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Read a document by ID"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (d:Document {id: $id})
                    RETURN d.text as text, d.metadata as metadata
                """, {"id": doc_id})
                record = result.single()
                if record:
                    return {
                        "text": record["text"],
                        "metadata": record["metadata"]
                    }
                return None
        except Exception as e:
            print(f"Error reading document: {str(e)}")
            return None

    def update_document(self, doc_id: str, text: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update document text and/or metadata"""
        try:
            updates = []
            params = {"id": doc_id}
            
            if text is not None:
                embedding = self.embeddings.embed_query(text)
                updates.extend([
                    "d.text = $text",
                    "d.embedding = $embedding"
                ])
                params.update({
                    "text": text,
                    "embedding": embedding
                })
            
            if metadata is not None:
                # Update timestamp
                metadata["updated_at"] = str(int(datetime.now().timestamp()))
                updates.append("d.metadata = $metadata")
                params["metadata"] = metadata

            if updates:
                with self.driver.session(database=self.database) as session:
                    result = session.run(f"""
                        MATCH (d:Document {{id: $id}})
                        SET {', '.join(updates)}
                        RETURN d.id as id
                    """, params)
                    return result.single() is not None
            return False
        except Exception as e:
            print(f"Error updating document: {str(e)}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its relationships"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (d:Document {id: $id})
                    DETACH DELETE d
                    RETURN count(d) as count
                """, {"id": doc_id})
                return result.single()["count"] > 0
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False

    def bulk_create_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Create multiple documents efficiently"""
        try:
            doc_ids = []
            current_time = str(int(datetime.now().timestamp()))
            
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else {}
                doc_id = self.create_document(text, metadata)
                if doc_id:
                    doc_ids.append(doc_id)
            
            if doc_ids:
                self.create_graph_relationships()
            
            return doc_ids
        except Exception as e:
            print(f"Error in bulk document creation: {str(e)}")
            return []

    def search_documents(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                    YIELD node, score
                    RETURN node.text as text,
                           node.metadata as metadata,
                           score
                    ORDER BY score DESC
                """, {
                    "index_name": self.index_name,
                    "embedding": query_embedding,
                    "k": k
                })
                
                return [{
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "score": record["score"]
                } for record in result]
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (d:Document)
                    RETURN count(d) as count
                """)
                return result.single()["count"]
        except Exception as e:
            print(f"Error getting document count: {str(e)}")
            return 0

    def get_documents_by_metadata(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """Get documents by metadata field"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.metadata[$key] = $value
                    RETURN d.id as id, d.text as text, d.metadata as metadata
                """, {"key": key, "value": value})
                
                return [{
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record["metadata"]
                } for record in result]
        except Exception as e:
            print(f"Error getting documents by metadata: {str(e)}")
            return []

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()