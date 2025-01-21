from typing import List, Dict, Any
from langchain_neo4j import Neo4jVector
from llama_index.embeddings import OllamaEmbedding
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class Neo4jHelper:
    def __init__(self):
        self.url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        # Initialize Ollama embeddings with llama2 model
        self.embeddings = OllamaEmbedding(
            model_name="llama3.3:70b",
            base_url="http://localhost:11434"
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
    
    def create_graph_relationships(self):
        """Create relationships between similar documents using cosine similarity"""
        query = """
        CALL gds.graph.project(
            'doc_graph',
            'Document',
            '*',
            {
                nodeProperties: ['embedding']
            }
        )
        YIELD graphName;

        CALL gds.nodeSimilarity.stream('doc_graph', {
            nodeProperties: ['embedding'],
            similarityCutoff: 0.7
        })
        YIELD node1, node2, similarity
        MATCH (d1:Document) WHERE id(d1) = node1
        MATCH (d2:Document) WHERE id(d2) = node2
        MERGE (d1)-[r:SIMILAR {score: similarity}]->(d2);

        CALL gds.graph.drop('doc_graph');
        """
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query)
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
            query_embedding = self.embeddings.get_query_embedding(query)
            
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