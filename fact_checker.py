import os
import yaml
from huggingface_hub import InferenceClient

class FactChecker:
    """
    A class to fact-check text using an LLM model with system and user prompts.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the FactChecker with an LLM model and prompts.
        Can load from config file or accept parameters directly.
        
        Args:
            model (str, optional): The model name/identifier. If None, loads from config.
            system_prompt (str, optional): The system prompt. If None, loads from config.
            user_prompt (str, optional): The user prompt template. If None, loads from config.
            config_path (str): Path to the config YAML file. Defaults to "config.yaml".
        """
        config = self._load_config(config_path)
        self.model = config.get("model", "openai/gpt-oss-120b")
        self.system_prompt = config.get("system_prompt", "You are a fact checker. You are given a text and you need to fact check it.")
        self.user_prompt = config.get("user_prompt", "Fact check the following text: {text}")
    
    @staticmethod
    def _load_config(config_path="config.yaml"):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the config YAML file.
            
        Returns:
            dict: Configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Config loaded successfully: {config}")
            return config
        except Exception as e:
            print(f"Error loading config, using default values: {e}")
            return {}
    
    def check_fact(self, text_to_check):
        """
        Fact-check the given text using the LLM model.
        
        Args:
            text_to_check (str): The text that needs to be fact-checked
            
        Returns:
            str: The fact-checking response from the LLM
        """
        # Format the user prompt with the text to check
        formatted_user_prompt = self.user_prompt.format(text=text_to_check)
        
        # Call the LLM with system and user prompts
        llm_response = self._call_llm(formatted_user_prompt)
        
        # Parse and return the response
        return self._parse_response(llm_response)
    
    def _call_llm(self, formatted_user_prompt):
        """
        Make the actual LLM API call with the given prompts.
        
        Args:
            formatted_user_prompt (str): The formatted user prompt
            
        Returns:
            str: The LLM response
        """
        client = InferenceClient()

        # Prepare messages with system and user prompts
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": formatted_user_prompt
            }]

        print(messages)
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return completion.choices[0].message.content
    
    def _parse_response(self, llm_response):
        """
        Parse and process the LLM response.
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            str: The processed fact-checking result
        """
        return llm_response

