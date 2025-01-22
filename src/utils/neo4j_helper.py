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
import numpy as np
from sklearn.decomposition import PCA

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
        self.target_dimensions = 4096  # Neo4j's maximum supported dimensions
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            dimensions=4096,  # Set fixed dimensions for Neo4j compatibility
            model=os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.3:70b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
        
        # Initialize PCA for dimensionality reduction
        self.pca = None
        
        # Initialize vector store and cache
        self._init_vector_store()

    def _reduce_dimensions(self, embedding: List[float]) -> List[float]:
        """Reduce embedding dimensions to match Neo4j's maximum"""
        if len(embedding) <= self.target_dimensions:
            return embedding
            
        # Convert to numpy array
        embedding_array = np.array(embedding).reshape(1, -1)
        
        # Initialize PCA if not already done
        if self.pca is None:
            self.pca = PCA(n_components=self.target_dimensions)
            reduced = self.pca.fit_transform(embedding_array)
        else:
            reduced = self.pca.transform(embedding_array)
        
        return reduced[0].tolist()

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten metadata to primitive types that Neo4j can store"""
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                flattened[key] = str(value)
            elif isinstance(value, bool):
                flattened[key] = str(value).lower()
            elif isinstance(value, (list, dict)):
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = str(value)
        return flattened

    def check_gds_plugin(self) -> bool:
        """Check if the Graph Data Science plugin is installed and available."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL gds.version() YIELD gdsVersion")
                version = result.single()["gdsVersion"]
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
            # Check if vector index exists and drop it
            with self.driver.session(database=self.database) as session:
                # First, try to drop the existing index if it exists
                session.run("""
                    DROP INDEX accounting_docs IF EXISTS
                """)
                
                # Create vector index using new syntax with correct dimensions
                session.run("""
                    CREATE VECTOR INDEX accounting_docs IF NOT EXISTS
                    FOR (n:Document)
                    ON (n.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 4096,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
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
                        # Get embedding and reduce dimensions
                        embedding = self.embeddings.embed_query(text)
                        # reduced_embedding = self._reduce_dimensions(embedding)
                        
                        session.run("""
                            MATCH (d:Document {id: $id})
                            SET d.embedding = $embedding
                        """, id=doc_id, embedding=embedding)
                    print("Vectorization complete")
        except Exception as e:
            print(f"Error vectorizing documents: {str(e)}")

    def create_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Create a single document with vector embedding"""
        try:
            doc_id = str(uuid.uuid4())
            # Get embedding and reduce dimensions
            embedding = self.embeddings.embed_query(text)
            # reduced_embedding = self._reduce_dimensions(embedding)
            
            if metadata is None:
                metadata = {}
            
            # Add basic metadata
            metadata.update({
                "id": doc_id,
                "created_at": str(int(datetime.now().timestamp())),
                "updated_at": str(int(datetime.now().timestamp()))
            })
            
            # Flatten metadata for Neo4j storage
            flattened_metadata = self._flatten_metadata(metadata)
            
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    CREATE (d:Document)
                    SET d.id = $id,
                        d.text = $text,
                        d.embedding = $embedding,
                        d += $metadata
                    RETURN d.id as id
                """, {
                    "id": doc_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": flattened_metadata
                })
                return result.single()["id"]
        except Exception as e:
            print(f"Error creating document: {str(e)}")
            return None

    def update_document(self, doc_id: str, text: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update document text and/or metadata"""
        try:
            updates = []
            params = {"id": doc_id}
            
            if text is not None:
                # Get embedding and reduce dimensions
                embedding = self.embeddings.embed_query(text)
                # reduced_embedding = self._reduce_dimensions(embedding)
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
                # Flatten metadata
                flattened_metadata = self._flatten_metadata(metadata)
                updates.append("d += $metadata")
                params["metadata"] = flattened_metadata

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

    def similarity_search_with_graph(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity with graph traversal"""
        try:
            # Get embedding and reduce dimensions
            query_embedding = self.embeddings.embed_query(query)
            # reduced_query_embedding = self._reduce_dimensions(query_embedding)
            
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                    YIELD node, score
                    WITH node, score
                    
                    OPTIONAL MATCH (node)-[sim:SIMILAR]->(similar:Document)
                    OPTIONAL MATCH (node)-[rel:RELATED_CONTENT]->(related:Document)
                    OPTIONAL MATCH (node)-[temp:TEMPORAL]->(temporal:Document)
                    OPTIONAL MATCH (node)-[:SHARES_TYPE]->(typeRelated:Document)
                    
                    RETURN 
                        node.text AS text,
                        score AS vector_score,
                        node {.*, embedding: null} AS metadata,
                        collect(DISTINCT {
                            text: similar.text,
                            score: sim.score,
                            type: 'similar'
                        }) AS similar_docs,
                        collect(DISTINCT {
                            text: related.text,
                            relevance: rel.relevance,
                            type: 'content'
                        }) AS related_docs,
                        collect(DISTINCT {
                            text: temporal.text,
                            time_diff: temp.time_diff,
                            type: 'temporal'
                        }) AS temporal_docs,
                        collect(DISTINCT {
                            text: typeRelated.text,
                            type: 'same_type'
                        }) AS type_docs
                """, {
                    "index_name": self.index_name,
                    "embedding": query_embedding,
                    "k": k
                })
                
                documents = []
                for record in result:
                    doc = {
                        "text": record["text"],
                        "score": record["vector_score"],
                        "metadata": record["metadata"],
                        "relationships": {
                            "similar": [d for d in record["similar_docs"] if d["text"]],
                            "content": [d for d in record["related_docs"] if d["text"]],
                            "temporal": [d for d in record["temporal_docs"] if d["text"]],
                            "type": [d for d in record["type_docs"] if d["text"]]
                        }
                    }
                    documents.append(doc)
                
                return documents
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []

    def read_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Read a document by ID"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (d:Document {id: $id})
                    RETURN d
                """, {"id": doc_id})
                record = result.single()
                if record:
                    node = record["d"]
                    return {
                        "id": node["id"],
                        "text": node["text"],
                        "metadata": {k: v for k, v in node.items() 
                                  if k not in ["id", "text", "embedding"]}
                    }
                return None
        except Exception as e:
            print(f"Error reading document: {str(e)}")
            return None

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

    def create_graph_relationships(self):
        """Create relationships between documents using multiple criteria"""
        try:
            if not self.check_gds_plugin():
                print("Neo4j Graph Data Science plugin is not available.")
                return False

            with self.driver.session(database=self.database) as session:
                # Create vector similarity relationships
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE elementId(d1) < elementId(d2)
                WITH d1, d2, gds.similarity.cosine(d1.embedding, d2.embedding) AS similarity
                WHERE similarity > 0.7
                CREATE (d1)-[:SIMILAR {score: similarity}]->(d2)
                """)

                # Create content-based relationships
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE elementId(d1) < elementId(d2)
                WITH d1, d2,
                    gds.similarity.cosine(d1.embedding, d2.embedding) AS contentSimilarity
                WHERE contentSimilarity > 0.3
                CREATE (d1)-[:RELATED_CONTENT {relevance: contentSimilarity}]->(d2)
                """)

                # Create temporal relationships
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE elementId(d1) < elementId(d2)
                AND d1.created_at IS NOT NULL 
                AND d2.created_at IS NOT NULL
                WITH d1, d2,
                    CASE
                        WHEN d1.created_at IS NOT NULL 
                            AND d2.created_at IS NOT NULL
                        THEN abs(toInteger(d1.created_at) - toInteger(d2.created_at)) / 86400
                        ELSE 999999
                    END AS daysDiff
                WHERE daysDiff <= 30
                CREATE (d1)-[:TEMPORAL {time_diff: daysDiff}]->(d2)
                """)

                # Create file type relationships
                session.run("""
                MATCH (d1:Document)
                MATCH (d2:Document)
                WHERE d1.file_type = d2.file_type AND elementId(d1) < elementId(d2)
                CREATE (d1)-[:SHARES_TYPE]->(d2)
                """)

            return True
        except Exception as e:
            print(f"Error creating relationships: {str(e)}")
            return False

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
                    WHERE d[$key] = $value
                    RETURN d
                """, {"key": key, "value": value})
                
                return [{
                    "id": record["d"]["id"],
                    "text": record["d"]["text"],
                    "metadata": {k: v for k, v in record["d"].items() 
                               if k not in ["id", "text", "embedding"]}
                } for record in result]
        except Exception as e:
            print(f"Error getting documents by metadata: {str(e)}")
            return []

    def safe_drop_graph(self, graph_name: str):
        """Safely drop a projected graph if it exists"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                    CALL gds.graph.exists($name)
                    YIELD exists
                    WITH exists
                    WHERE exists
                    CALL gds.graph.drop($name)
                    YIELD graphName
                    RETURN graphName
                """, {"name": graph_name})
        except Exception as e:
            print(f"Error dropping graph {graph_name}: {str(e)}")

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()