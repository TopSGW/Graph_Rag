import pytest
from unittest.mock import Mock, patch, MagicMock
from neo4j import GraphDatabase, Driver, Session, Transaction, Result
from src.utils.neo4j_helper import Neo4jHelper

class TestNeo4jHelper:
    @pytest.fixture
    def mock_driver(self):
        driver = Mock(spec=Driver)
        session = Mock(spec=Session)
        transaction = Mock(spec=Transaction)
        result = Mock(spec=Result)
        
        # Setup mock returns
        result.data.return_value = [{"n": {"name": "test"}}]
        result.single.return_value = {
            "version": "2.4.0",
            "graphName": "doc_graph",
            "nodeCount": 10,
            "relationshipCount": 20,
            "projectMillis": 100,
            "nodesCompared": 45,
            "relationshipsWritten": 15,
            "computeMillis": 200,
            "similarityDistribution": {"min": 0.7, "max": 0.9, "mean": 0.8}
        }
        transaction.run.return_value = result
        session.begin_transaction.return_value = transaction
        session.run.return_value = result
        driver.session.return_value = session
        
        return driver

    @pytest.fixture
    def neo4j_helper(self):
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            helper = Neo4jHelper()
            yield helper

    def test_initialization(self, neo4j_helper):
        """Test Neo4jHelper initialization"""
        assert isinstance(neo4j_helper, Neo4jHelper)
        assert neo4j_helper.url == "neo4j://localhost:7687"
        assert neo4j_helper.username == "neo4j"
        assert neo4j_helper.password == "neo4j"
        assert neo4j_helper.database == "neo4j"

    def test_check_gds_plugin(self, neo4j_helper, mock_driver):
        """Test GDS plugin check"""
        with patch.object(neo4j_helper, 'driver', mock_driver):
            assert neo4j_helper.check_gds_plugin() == True
            mock_driver.session().run.assert_called_with("RETURN gds.version() AS version")

    def test_check_gds_plugin_failure(self, neo4j_helper, mock_driver):
        """Test GDS plugin check failure"""
        mock_driver.session().run.side_effect = Exception("GDS not available")
        with patch.object(neo4j_helper, 'driver', mock_driver):
            assert neo4j_helper.check_gds_plugin() == False

    @patch('langchain_neo4j.Neo4jVector')
    def test_initialize_vector_store(self, mock_vector, neo4j_helper):
        """Test vector store initialization"""
        mock_vector.from_existing_index.return_value = MagicMock()
        mock_vector.from_documents.return_value = MagicMock()

        # Test initialization without documents
        vector_store = neo4j_helper.initialize_vector_store()
        mock_vector.from_existing_index.assert_called_once()

        # Test initialization with documents
        documents = [{"page_content": "test", "metadata": {"source": "test"}}]
        vector_store = neo4j_helper.initialize_vector_store(documents)
        mock_vector.from_documents.assert_called_once()

    def test_create_graph_relationships(self, neo4j_helper, mock_driver):
        """Test graph relationship creation"""
        with patch.object(neo4j_helper, 'driver', mock_driver):
            # First ensure GDS check passes
            mock_driver.session().run.return_value.single.return_value = {"version": "2.4.0"}
            
            assert neo4j_helper.create_graph_relationships() == True
            
            # Verify the sequence of calls
            calls = mock_driver.session().run.call_args_list
            assert len(calls) >= 4  # Should have at least 4 calls: GDS check, drop, project, similarity, drop
            
            # Verify the graph projection call
            project_call = [call for call in calls if "gds.graph.project" in str(call)][0]
            assert "nodeProperties" in str(project_call)
            assert "embedding" in str(project_call)
            
            # Verify the node similarity call
            similarity_call = [call for call in calls if "gds.nodeSimilarity.write" in str(call)][0]
            assert "writeRelationshipType" in str(similarity_call)
            assert "similarityCutoff" in str(similarity_call)
            assert "topK" in str(similarity_call)

    def test_create_graph_relationships_failure(self, neo4j_helper, mock_driver):
        """Test graph relationship creation failure"""
        mock_driver.session().run.side_effect = Exception("Failed to create relationships")
        with patch.object(neo4j_helper, 'driver', mock_driver):
            assert neo4j_helper.create_graph_relationships() == False

    def test_similarity_search_with_graph(self, neo4j_helper, mock_driver):
        """Test similarity search with graph enhancement"""
        mock_result = [
            {
                "text": "test document",
                "score": 0.9,
                "metadata": {
                    "source": "test",
                    "related_documents": [
                        {"text": "related doc", "similarity": 0.8}
                    ]
                }
            }
        ]
        mock_driver.session().run.return_value.data.return_value = mock_result

        with patch.object(neo4j_helper, 'driver', mock_driver):
            with patch.object(neo4j_helper, 'initialize_vector_store'):
                with patch.object(neo4j_helper.embeddings, 'embed_query', return_value=[0.1] * 4096):
                    results = neo4j_helper.similarity_search_with_graph("test query", k=1)
                    assert results == mock_result
                    assert len(mock_driver.session().run.call_args_list) > 0

    def test_add_documents(self, neo4j_helper):
        """Test document addition with graph relationships"""
        with patch.object(neo4j_helper, 'initialize_vector_store') as mock_init:
            with patch.object(neo4j_helper, 'create_graph_relationships') as mock_create:
                mock_create.return_value = True
                
                texts = ["doc1", "doc2"]
                metadatas = [{"source": "src1"}, {"source": "src2"}]
                
                assert neo4j_helper.add_documents(texts, metadatas) == True
                mock_init.assert_called_once()
                mock_create.assert_called_once()

    def test_add_documents_failure(self, neo4j_helper):
        """Test document addition failure"""
        with patch.object(neo4j_helper, 'initialize_vector_store', side_effect=Exception("Failed to add documents")):
            texts = ["doc1", "doc2"]
            assert neo4j_helper.add_documents(texts) == False

    def test_close(self, neo4j_helper, mock_driver):
        """Test connection closing"""
        with patch.object(neo4j_helper, 'driver', mock_driver):
            neo4j_helper.close()
            mock_driver.close.assert_called_once()