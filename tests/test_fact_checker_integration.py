"""Integration tests for FactChecker class."""
import pytest
from unittest.mock import patch, MagicMock
from fact_checker import FactChecker
from constants import *

TEST_PROMPT_TEMPLATE = "Fact check: {text}"

class TestCheckFactIntegration:
    """Integration tests for the check_fact method."""

    @staticmethod
    def _get_config(deployment_type="inference_client", wiki_agentic_rag=False, izzyviz=False):
        """
        Helper function to get test config.

        Args:
            deployment_type: Type of deployment ("inference_client" or "local")
            wiki_agentic_rag: Whether wiki agentic RAG is enabled
            izzyviz: Whether izzyviz is enabled

        Returns:
            dict: Configuration dictionary for testing
        """
        config = {
            "deployment_type": deployment_type,
            "model": "test-model",
            "temperature": 0,
            "wiki_agentic_rag": wiki_agentic_rag,
            "system_prompt_path": "prompts/system_prompt.txt",
            "user_prompt_path": "prompts/user_prompt.txt"
        }
        if deployment_type == "local" and izzyviz:
            config["izzyviz"] = izzyviz
        return config

    @staticmethod
    def _get_mock_inference_client():
        """
        Helper function to get mock inference client.

        Returns:
            MagicMock: Mock inference client
        """
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '<response>{"is_fact_true": true, "reasoning": "This is a verified fact"}</response>'
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        return mock_client

    @staticmethod
    def _get_mock_tokenizer_and_model(mock_response):
        """
        Helper function to get mock tokenizer and model.

        Returns:
            MagicMock: Mock tokenizer and model
        """
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.to.return_value = {
            'input_ids': MagicMock()
        }
        mock_tokenizer.decode.return_value = mock_response

        mock_model = MagicMock()
        mock_model.device = 'cpu'
        mock_outputs = MagicMock()
        mock_outputs.__getitem__.return_value = MagicMock()
        mock_model.generate.return_value = mock_outputs
        return mock_tokenizer, mock_model

    @staticmethod
    def _get_mock_agent(mock_agent_response):
        """
        Helper function to get mock agent.

        Returns:
            MagicMock: Mock agent
        """
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_agent_response
        return mock_agent

    @staticmethod
    def _get_mock_client_that_raises_exception():
        """
        Helper function to get mock client that raises an exception.

        Returns:
            MagicMock: Mock client that raises an exception
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        return mock_client

    @patch('fact_checker.FactChecker.load_config')
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.InferenceClient')
    def test_check_fact_with_inference_client(self, mock_inference_client_class,
                                              mock_load_prompt, mock_load_config):
        """Test full fact checking flow with inference client."""
        # Mock load config and prompt template
        mock_load_config.return_value = TestCheckFactIntegration._get_config(deployment_type="inference_client", wiki_agentic_rag=False, izzyviz=False)
        mock_load_prompt.return_value = TEST_PROMPT_TEMPLATE
        
        # Mock the client 
        mock_client = TestCheckFactIntegration._get_mock_inference_client()
        mock_inference_client_class.return_value = mock_client
        
        # Create fact checker and test
        fact_checker = FactChecker(config_path="dummy.yaml")
        result = fact_checker.check_fact("The Earth is round.")
        
        # Assertions
        assert result is not None
        assert result["is_fact_true"] is True
        assert result["reasoning"] == "This is a verified fact"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == SYSTEM_ROLE
        assert call_args[1]["messages"][1]["role"] == USER_ROLE
    
    @patch('fact_checker.FactChecker.load_config')
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.AutoModelForCausalLM')
    @patch('fact_checker.AutoTokenizer')
    def test_check_fact_with_local_model(self, mock_tokenizer_class, mock_model_class,
                                        mock_load_prompt, mock_load_config):
        """Test full fact checking flow with local model."""
        # Mock load config and prompt template
        mock_load_config.return_value = TestCheckFactIntegration._get_config(deployment_type="local", wiki_agentic_rag=False, izzyviz=False)
        mock_load_prompt.return_value = TEST_PROMPT_TEMPLATE
        
        # Mock tokenizer and model
        mock_response = '<response>{"is_fact_true": false, "reasoning": "This fact is incorrect"}</response>'
        mock_tokenizer_class.from_pretrained.return_value, mock_model_class.from_pretrained.return_value = TestCheckFactIntegration._get_mock_tokenizer_and_model(mock_response)
        
        # Create fact checker and test
        fact_checker = FactChecker(config_path="dummy.yaml")
        result = fact_checker.check_fact("The moon is made of cheese.")
        
        # Assertions
        assert result is not None
        assert result["is_fact_true"] is False
        assert result["reasoning"] == "This fact is incorrect"
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('fact_checker.FactChecker.load_config')
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.CodeAgent')
    @patch('fact_checker.InferenceClientModel')
    @patch('fact_checker.WikipediaSearchTool')
    def test_check_fact_with_agentic_rag(self, mock_wiki_tool, mock_model, 
                                         mock_agent_class, mock_load_prompt, 
                                         mock_load_config):
        """Test full fact checking flow with agentic RAG."""
        # Mock load config and prompt template
        mock_load_config.return_value = TestCheckFactIntegration._get_config(wiki_agentic_rag=True)
        mock_load_prompt.return_value = TEST_PROMPT_TEMPLATE
        
        # Mock agent
        mock_agent_response = '<response>{"is_fact_true": true, "reasoning": "Verified using Wikipedia"}</response>'
        mock_agent = TestCheckFactIntegration._get_mock_agent(mock_agent_response)
        mock_agent_class.return_value = mock_agent
        
        # Create fact checker and test
        fact_checker = FactChecker(config_path="dummy.yaml")
        result = fact_checker.check_fact("Python is a programming language.")
        
        # Assertions
        assert result is not None
        assert result["is_fact_true"] is True
        assert result["reasoning"] == "Verified using Wikipedia"
        mock_agent.run.assert_called_once()
    
    @patch('fact_checker.FactChecker.load_config')
    @patch('fact_checker.FactChecker.load_prompt_template')
    @patch('fact_checker.InferenceClient')
    def test_check_fact_handles_llm_error(self, mock_inference_client_class,
                                         mock_load_prompt, mock_load_config):
        """Test that check_fact handles LLM errors gracefully."""
        # Mock load config and prompt template
        mock_load_config.return_value = TestCheckFactIntegration._get_config(deployment_type="inference_client", wiki_agentic_rag=False, izzyviz=False)
        mock_load_prompt.return_value = TEST_PROMPT_TEMPLATE
        
        # Mock client to raise an exception
        mock_client = TestCheckFactIntegration._get_mock_client_that_raises_exception()
        mock_inference_client_class.return_value = mock_client

        # Create fact checker and test
        fact_checker = FactChecker(config_path="dummy.yaml")
        result = fact_checker.check_fact("Test fact")
        
        # Assertions : Should return None when LLM call fails
        assert result is None
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == SYSTEM_ROLE
        assert call_args[1]["messages"][1]["role"] == USER_ROLE