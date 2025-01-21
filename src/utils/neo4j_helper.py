from typing import List, Dict, Any
from langchain_neo4j import Neo4jVector
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase
from config.config import config

class Neo4jHelper:
    def __init__(self):
        neo4j_config = config.get_neo4j_config()
        ollama_config = config.get_ollama_config()
        
        self.url = neo4j_config['uri']
        self.username = neo4j_config['user']
        self.password = neo4j_config['password']
        self.database = neo4j_config['database']
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=ollama_config['default_model'],
            base_url=ollama_config['base_url']
        )
        self.driver = GraphDatabase.driver(self.url, auth=(self.username, self.password))
        
    def initialize_vector_store(self, documents: List[Dict[str, Any]] = None) -> Neo4jVector:
        """Initialize Neo4j Vector store with documents if provided"""
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
        """Check if the Graph Data Science plugin is installed and available"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN gds.version() AS version")
                version = result.single()["version"]
                print(f"GDS version: {version}")
                return True
        except Exception as e:
            print(f"Error checking GDS plugin: {str(e)}")
            return False
    
    def create_graph_relationships(self):
        """Create relationships between similar documents using cosine similarity"""
        try:
            # First check if GDS plugin is available
            if not self.check_gds_plugin():
                print("Neo4j Graph Data Science plugin is not available. Please install it first.")
                return False

            with self.driver.session(database=self.database) as session:
                # First check if the graph already exists and drop it if it does
                try:
                    session.run("CALL gds.graph.drop('doc_graph', false)")
                except:
                    pass  # Ignore error if graph doesn't exist

                # Step 1: Create a native projection
                project_query = """
                CALL gds.graph.project.cypher(
                    'doc_graph',
                    'MATCH (n:Document) RETURN id(n) AS id, n.embedding AS embedding',
                    'MATCH (n:Document)-[r:SIMILAR]->(m:Document) RETURN id(n) AS source, id(m) AS target, r.score AS score',
                    {
                        nodeProperties: ['embedding']
                    }
                )
                """
                session.run(project_query)

                # Step 2: Run node similarity
                similarity_query = """
                CALL gds.nodeSimilarity.write(
                    'doc_graph',
                    {
                        writeRelationshipType: 'SIMILAR',
                        writeProperty: 'score',
                        similarityCutoff: 0.7
                    }
                )
                YIELD nodesCompared, relationshipsWritten
                """
                result = session.run(similarity_query)
                stats = result.single()
                print(f"Nodes compared: {stats['nodesCompared']}, Relationships written: {stats['relationshipsWritten']}")

                # Step 3: Drop the projected graph
                session.run("CALL gds.graph.drop('doc_graph', false)")

            return True
        except Exception as e:
            print(f"Error creating relationships: {str(e)}")
            return False

    def similarity_search_with_graph(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform similarity search with graph-enhanced results
        Returns documents along with their related documents
        """
        retrieval_query = """
        WITH $query AS query
        CALL db.index.vector.queryNodes('accounting_docs', $k, query)
        YIELD node, score
        OPTIONAL MATCH (node)-[r:SIMILAR]->(related:Document)
        WITH node, score, collect({text: related.text, similarity: r.score}) as related_docs
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
        """Add documents to the vector store"""
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
        """Close the Neo4j driver connection"""
        self.driver.close()