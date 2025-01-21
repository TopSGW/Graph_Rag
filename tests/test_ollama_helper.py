import pytest
import json
from unittest.mock import Mock, patch
import requests
from src.utils.ollama_helper import OllamaHelper

class TestOllamaHelper:
    @pytest.fixture
    def ollama_helper(self):
        return OllamaHelper(base_url="http://localhost:11434")

    @pytest.fixture
    def mock_response(self):
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {
            "model": "llama2",
            "created_at": "2024-01-20T12:00:00.000Z",
            "response": "Test response",
            "done": True
        }
        return response

    def test_initialization(self):
        """Test OllamaHelper initialization"""
        helper = OllamaHelper(base_url="http://localhost:11434")
        assert isinstance(helper, OllamaHelper)
        assert helper.base_url == "http://localhost:11434"

    def test_generate_text(self, ollama_helper, mock_response):
        """Test text generation"""
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response
            
            response = ollama_helper.generate_text(
                prompt="Test prompt",
                model="llama2",
                temperature=0.7,
                max_tokens=100
            )
            
            assert response == "Test response"
            mock_post.assert_called_once()
            
            # Verify request payload
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/generate"
            assert json.loads(call_args[1]['data'])['prompt'] == "Test prompt"

    def test_generate_text_with_parameters(self, ollama_helper, mock_response):
        """Test text generation with various parameters"""
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response
            
            response = ollama_helper.generate_text(
                prompt="Test prompt",
                model="llama2",
                temperature=0.5,
                max_tokens=50,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            assert response == "Test response"
            
            # Verify all parameters were passed correctly
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            assert request_data['temperature'] == 0.5
            assert request_data['max_tokens'] == 50

    def test_error_handling(self, ollama_helper):
        """Test error handling in text generation"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("Connection Error")
            
            with pytest.raises(Exception) as exc_info:
                ollama_helper.generate_text("Test prompt", "llama2")
            assert "Connection Error" in str(exc_info.value)

    def test_invalid_response(self, ollama_helper):
        """Test handling of invalid API responses"""
        mock_invalid_response = Mock(spec=requests.Response)
        mock_invalid_response.status_code = 400
        mock_invalid_response.json.return_value = {"error": "Invalid request"}
        
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_invalid_response
            
            with pytest.raises(Exception) as exc_info:
                ollama_helper.generate_text("Test prompt", "llama2")
            assert "Invalid request" in str(exc_info.value)

    def test_model_loading(self, ollama_helper, mock_response):
        """Test model loading functionality"""
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response
            
            response = ollama_helper.load_model("llama2")
            assert response == mock_response.json.return_value
            
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/pull"
            assert json.loads(call_args[1]['data'])['name'] == "llama2"

    def test_streaming_response(self, ollama_helper):
        """Test handling of streaming responses"""
        mock_stream_response = Mock(spec=requests.Response)
        mock_stream_response.iter_lines.return_value = [
            json.dumps({"response": "Part 1", "done": False}).encode(),
            json.dumps({"response": "Part 2", "done": True}).encode()
        ]
        
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_stream_response
            
            response = ollama_helper.generate_text(
                prompt="Test prompt",
                model="llama2",
                stream=True
            )
            
            assert response == "Part 1Part 2"

    def test_context_handling(self, ollama_helper, mock_response):
        """Test handling of conversation context"""
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response
            
            context = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            
            response = ollama_helper.generate_text(
                prompt="Test prompt",
                model="llama2",
                context=context
            )
            
            assert response == "Test response"
            
            # Verify context was passed correctly
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            assert 'context' in request_data
            assert request_data['context'] == context

    def test_timeout_handling(self, ollama_helper):
        """Test handling of request timeouts"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            
            with pytest.raises(Exception) as exc_info:
                ollama_helper.generate_text(
                    prompt="Test prompt",
                    model="llama2",
                    timeout=5
                )
            assert "Request timed out" in str(exc_info.value)

    def test_connection_error_handling(self, ollama_helper):
        """Test handling of connection errors"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            with pytest.raises(Exception) as exc_info:
                ollama_helper.generate_text("Test prompt", "llama2")
            assert "Connection failed" in str(exc_info.value)