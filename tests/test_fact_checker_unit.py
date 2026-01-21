"""Unit tests for FactChecker class."""
import pytest
import json
import tempfile
import os
import yaml
from unittest.mock import patch, mock_open, MagicMock
from fact_checker import FactChecker
from constants import *

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config = {
        "deployment_type": "inference_client",
        "model": "test-model",
        "temperature": 0,
        "system_prompt_path": "prompts/system_prompt.txt",
        "user_prompt_path": "prompts/user_prompt.txt",
        "wiki_agentic_rag": False,
        "izzyviz": False
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def temp_prompt_file():
    """Create a temporary prompt file for testing."""
    content = "Test prompt template: {text}"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestLoadConfig:
    """Test the load_config static method."""
    
    def test_load_config_success(self, temp_config_file):
        """Test successful config loading."""
        config = FactChecker.load_config(temp_config_file)
        assert isinstance(config, dict)
        assert config["model"] == "test-model"
        assert config["deployment_type"] == "inference_client"
    
    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        config = FactChecker.load_config("nonexistent_config.yaml")
        assert config == {}


class TestLoadPromptTemplate:
    """Test the load_prompt_template static method."""
    
    def test_load_prompt_template_success(self, temp_prompt_file):
        """Test successful prompt template loading."""
        prompt = FactChecker.load_prompt_template(temp_prompt_file)
        assert prompt == "Test prompt template: {text}"
    
    def test_load_prompt_template_file_not_found(self):
        """Test default prompt is returned when prompt file doesn't exist."""
        prompt = FactChecker.load_prompt_template("nonexistent_prompt.txt")
        assert prompt == "Fact check the following text: {text}. Return JSON output with keys 'is_fact_true' and 'reasoning'"


class TestFactCheckerInitialization:
    """Test FactChecker initialization."""
    
    @patch('fact_checker.InferenceClient')
    @patch('fact_checker.FactChecker.load_config')
    @patch('fact_checker.FactChecker.load_prompt_template')
    def test_init_with_inference_client(self, mock_load_prompt, mock_load_config, 
                                        mock_inference_client):
        """Test initialization with inference client deployment type."""
        #Mock load_config, load_prompt_template
        mock_load_config.return_value = {
            "deployment_type": "inference_client",
            "model": "test-model",
            "temperature": 0,
            "wiki_agentic_rag": False,
            "izzyviz": True
        }
        mock_load_prompt.return_value = "Test prompt"
        mock_inference_client.return_value = MagicMock()
        
        #Initialize fact checker
        fact_checker = FactChecker(config_path=temp_config_file)
        
        assert fact_checker.deployment_type == "inference_client"
        assert fact_checker.model == "test-model"
        assert fact_checker.temperature is 0
        assert fact_checker.system_prompt == "Test prompt"
        assert fact_checker.user_prompt == "Test prompt"
        assert fact_checker.izzyviz is True
        assert fact_checker.client is not None
    
    @patch('fact_checker.AutoTokenizer')
    @patch('fact_checker.AutoModelForCausalLM')
    @patch('fact_checker.FactChecker.load_config')
    @patch('fact_checker.FactChecker.load_prompt_template')
    def test_init_with_local_deployment(self, mock_load_prompt, mock_load_config, 
                                       mock_model, mock_tokenizer):
        """Test initialization with local deployment type."""
        #Mock load_config, load_prompt_template, tokenizer from_pretrained and model from_pretrained
        mock_load_config.return_value = {
            "deployment_type": "local",
            "model": "test-model",
            "temperature": 0,
            "wiki_agentic_rag": False,
            "izzyviz": False
        }
        mock_load_prompt.return_value = "Test prompt"
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        #Initialize fact checker
        fact_checker = FactChecker(config_path="dummy.yaml")
        
        assert fact_checker.deployment_type == "local"
        assert fact_checker.loaded_tokenizer is not None
        assert fact_checker.loaded_model is not None
        assert fact_checker.client is None
    
    @patch('fact_checker.CodeAgent')
    @patch('fact_checker.InferenceClientModel')
    @patch('fact_checker.WikipediaSearchTool')
    @patch('fact_checker.FactChecker.load_config')
    @patch('fact_checker.FactChecker.load_prompt_template')
    def test_init_with_wiki_agentic_rag(self, mock_load_prompt, mock_load_config,
                                        mock_wiki_tool, mock_model, mock_agent):
        """Test initialization with wiki agentic RAG enabled."""
        #Mock load_config, load_prompt_template, CodeAgent
        mock_load_config.return_value = {
            "deployment_type": "inference_client",
            "model": "test-model",
            "temperature": 0,
            "wiki_agentic_rag": True,
            "izzyviz": False
        }
        mock_load_prompt.return_value = "Test prompt"
        mock_wiki_tool.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        mock_agent.return_value = MagicMock()
        
        fact_checker = FactChecker(config_path="dummy.yaml")
        
        assert fact_checker.wiki_agentic_rag is True
        assert fact_checker.model == "test-model"
        assert fact_checker.temperature is 0
        assert fact_checker.agent is not None


class TestParseResponse:
    """Test the _parse_response method."""
    
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.FactChecker._initialize_model_components')
    def test_parse_response_with_tags(self, mock_initialize_model_components, mock_load_prompt_template):
        """Test parsing response with <response> tags."""
        #Initialize fact checker with mocked components
        mock_initialize_model_components.return_value = None
        mock_load_prompt_template.return_value = None
        fact_checker = FactChecker(config_path=temp_config_file)
        
        #Parse response
        response = '<response>{"is_fact_true": true, "reasoning": "Valid fact"}</response>'
        result = fact_checker._parse_response(response)
        
        assert result is not None
        assert result["is_fact_true"] is True
        assert result["reasoning"] == "Valid fact"
    
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.FactChecker._initialize_model_components')
    def test_parse_response_without_tags(self, mock_initialize_model_components, mock_load_prompt_template):
        """Test parsing response without tags."""
        #Initialize fact checker with mocked components
        mock_initialize_model_components.return_value = None
        mock_load_prompt_template.return_value = None
        fact_checker = FactChecker(config_path=temp_config_file)
        
        #Parse response
        response = '{"is_fact_true": false, "reasoning": "Invalid fact"}'
        result = fact_checker._parse_response(response)
        
        assert result is not None
        assert result["is_fact_true"] is False
        assert result["reasoning"] == "Invalid fact"
    
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.FactChecker._initialize_model_components')
    def test_parse_response_dict_input(self, mock_initialize_model_components, mock_load_prompt_template):
        """Test parsing when response is already a dict."""
        #Initialize fact checker with mocked components
        mock_initialize_model_components.return_value = None
        mock_load_prompt_template.return_value = None
        fact_checker = FactChecker(config_path=temp_config_file)
        
        #Parse response
        response = {"is_fact_true": True, "reasoning": "Test"}
        result = fact_checker._parse_response(response)
        
        assert result is not None
        assert result["is_fact_true"] is True
        assert result["reasoning"] == "Test"
    
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.FactChecker._initialize_model_components')
    def test_parse_response_none(self, mock_initialize_model_components, mock_load_prompt_template):
        """Test parsing None response."""
        #Initialize fact checker with mocked components
        mock_initialize_model_components.return_value = None
        mock_load_prompt_template.return_value = None
        fact_checker = FactChecker(config_path=temp_config_file)
        
        #Parse response
        result = fact_checker._parse_response(None)

        assert result is None
    
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.FactChecker._initialize_model_components')
    def test_parse_response_missing_keys(self, mock_initialize_model_components, mock_load_prompt_template):
        """Test parsing response with missing keys."""
        #Initialize fact checker with mocked components
        mock_initialize_model_components.return_value = None
        mock_load_prompt_template.return_value = None
        fact_checker = FactChecker(config_path=temp_config_file)
        
        #Parse response
        response = '{"other_key": "value"}'
        result = fact_checker._parse_response(response)
        
        assert result is None