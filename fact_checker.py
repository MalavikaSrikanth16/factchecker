import os
import yaml
import json
import re
from huggingface_hub import InferenceClient

class FactChecker:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the FactChecker from a config file and prompt template files.
        
        Args:
            config_path (str): Path to the config YAML file. Defaults to "config.yaml".
        """
        config = self._load_config(config_path)
        self.model = config.get("model", "openai/gpt-oss-120b")
        self.temperature = config.get("temperature", 0)
        self.system_prompt = self._load_prompt_template(config.get("system_prompt_path", "prompts/system_prompt.txt"))
        self.user_prompt = self._load_prompt_template(config.get("user_prompt_path", "prompts/user_prompt.txt"))

    @staticmethod
    def _load_prompt_template(prompt_path):
        """
        Load a prompt template from a file.
        
        Args:
            prompt_path (str): Path to the prompt template file.
            
        Returns:
            str: The prompt template string, or empty string if file not found.
        """
        try:
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading prompt template from {prompt_path}: {e}")
            return ""
    
    @staticmethod
    def _load_config(config_path):
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
            dict: Dictionary containing 'is_fact_true' and 'reasoning' keys,
                  or None if parsing fails
        """
        # Format the user prompt template with the text to check
        formatted_user_prompt = self.user_prompt.replace("{text}", text_to_check)
        
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
            temperature=self.temperature,
        )
        return completion.choices[0].message.content
    
    def _parse_response(self, llm_response):
        """
        Parse and process the LLM response to extract is_fact_true and reasoning.
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            dict: Dictionary containing 'is_fact_true' and 'reasoning' keys,
                  or None if parsing fails
        """
        try:
            # Extract content between <response> and </response> tags
            match = re.search(r'<response>(.*?)</response>', llm_response, re.DOTALL)
            if not match:
                print(f"Warning: Could not find <response> tags in response: {llm_response}")
                return None
            
            json_str = match.group(1).strip()
            
            # Load the JSON
            parsed_data = json.loads(json_str)
            
            # Extract is_fact_true and reasoning
            result = {
                "is_fact_true": parsed_data.get("is_fact_true"),
                "reasoning": parsed_data.get("reasoning")
            }
            
            return result
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

