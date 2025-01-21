import pytest
from unittest.mock import Mock, patch
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
        transaction.run.return_value = result
        session.begin_transaction.return_value = transaction
        driver.session.return_value = session
        
        return driver

    @pytest.fixture
    def neo4j_helper(self):
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            helper = Neo4jHelper(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            yield helper

    def test_initialization(self):
        """Test Neo4jHelper initialization"""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            helper = Neo4jHelper(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            )
            assert isinstance(helper, Neo4jHelper)
            mock_driver.assert_called_once()

    def test_connection(self, neo4j_helper, mock_driver):
        """Test database connection"""
        with patch.object(neo4j_helper, '_driver', mock_driver):
            assert neo4j_helper.test_connection() == True

    def test_failed_connection(self, neo4j_helper):
        """Test failed database connection"""
        with patch.object(neo4j_helper, '_driver', side_effect=Exception):
            assert neo4j_helper.test_connection() == False

    def test_execute_query(self, neo4j_helper, mock_driver):
        """Test query execution"""
        test_query = "MATCH (n) RETURN n LIMIT 1"
        with patch.object(neo4j_helper, '_driver', mock_driver):
            result = neo4j_helper.execute_query(test_query)
            assert result == [{"n": {"name": "test"}}]
            mock_driver.session.assert_called_once()

    def test_execute_query_with_parameters(self, neo4j_helper, mock_driver):
        """Test query execution with parameters"""
        test_query = "MATCH (n) WHERE n.name = $name RETURN n"
        test_params = {"name": "test"}
        
        with patch.object(neo4j_helper, '_driver', mock_driver):
            result = neo4j_helper.execute_query(test_query, test_params)
            assert result == [{"n": {"name": "test"}}]
            
            # Verify parameter passing
            transaction = mock_driver.session().begin_transaction()
            transaction.run.assert_called_once_with(test_query, test_params)

    def test_execute_query_error(self, neo4j_helper, mock_driver):
        """Test error handling in query execution"""
        test_query = "INVALID QUERY"
        mock_driver.session().begin_transaction().run.side_effect = Exception("Query Error")
        
        with patch.object(neo4j_helper, '_driver', mock_driver):
            with pytest.raises(Exception) as exc_info:
                neo4j_helper.execute_query(test_query)
            assert "Query Error" in str(exc_info.value)

    def test_close_connection(self, neo4j_helper, mock_driver):
        """Test connection closing"""
        with patch.object(neo4j_helper, '_driver', mock_driver):
            neo4j_helper.close()
            mock_driver.close.assert_called_once()

    def test_context_manager(self):
        """Test context manager functionality"""
        with patch('neo4j.GraphDatabase.driver') as mock_driver:
            with Neo4jHelper(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="password"
            ) as helper:
                assert isinstance(helper, Neo4jHelper)
            mock_driver().close.assert_called_once()

    def test_batch_query_execution(self, neo4j_helper, mock_driver):
        """Test batch query execution"""
        test_queries = [
            ("MATCH (n:Label1) RETURN n", {}),
            ("MATCH (n:Label2) RETURN n", {"param": "value"})
        ]
        
        with patch.object(neo4j_helper, '_driver', mock_driver):
            results = neo4j_helper.execute_batch_queries(test_queries)
            assert len(results) == 2
            assert all(result == [{"n": {"name": "test"}}] for result in results)

    def test_transaction_management(self, neo4j_helper, mock_driver):
        """Test transaction management"""
        with patch.object(neo4j_helper, '_driver', mock_driver):
            with neo4j_helper.transaction() as tx:
                result = tx.run("MATCH (n) RETURN n")
                assert result.data() == [{"n": {"name": "test"}}]

    def test_database_error_handling(self, neo4j_helper, mock_driver):
        """Test database error handling"""
        mock_driver.session.side_effect = Exception("Database Error")
        
        with patch.object(neo4j_helper, '_driver', mock_driver):
            with pytest.raises(Exception) as exc_info:
                neo4j_helper.execute_query("MATCH (n) RETURN n")
            assert "Database Error" in str(exc_info.value)