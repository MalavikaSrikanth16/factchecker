import os
import yaml
import json
import re
import logging
from huggingface_hub import InferenceClient
from smolagents import CodeAgent, InferenceClientModel, WikipediaSearchTool

logger = logging.getLogger(__name__)

class FactChecker:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the FactChecker from a config file.
        
        Args:
            config_path (str): Path to the config YAML file. Defaults to "config.yaml".
        """
        config = self._load_config(config_path)
        self.model = config.get("model", "openai/gpt-oss-120b")
        self.temperature = config.get("temperature", 0)
        self.wiki_agentic_rag = config.get("wiki_agentic_rag", False)
        self.system_prompt = self._load_prompt_template(config.get("system_prompt_path", "prompts/system_prompt.txt"))
        self.user_prompt = self._load_prompt_template(config.get("user_prompt_path", "prompts/user_prompt.txt"))
        
        # Initialize agent with Wikipedia search tool if wiki_agentic_rag is enabled
        self.agent = None
        if self.wiki_agentic_rag:
            wiki_tool = WikipediaSearchTool(
                user_agent="FactChecker/1.0 (fact-checking-tool)",
                language="en",
                content_type="text",
                extract_format="WIKI"
            )
            agent_model = InferenceClientModel(model_id=self.model, temperature=self.temperature)
            self.agent = CodeAgent(tools=[wiki_tool], model=agent_model)

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
            logger.warning(f"Error loading prompt template from {prompt_path}: {e}")
            return "Fact check the following text: {text}. Return JSON output with keys 'is_fact_true' and 'reasoning'"
    
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
            logger.info(f"Config loaded successfully: {config}")
            return config
        except Exception as e:
            logger.warning(f"Error loading config, will fallback to default values: {e}")
            return {}
    
    def check_fact(self, text_to_check):
        """
        Fact-check the given text using the LLM.
        If wiki_agentic_rag is true, uses agentic RAG with Wikipedia search tool.
        
        Args:
            text_to_check (str): The text that needs to be fact-checked
            
        Returns:
            dict or str: Dictionary containing 'is_fact_true' and 'reasoning' keys if parsing succeeds,
                        or None if no suitable result is obtained
        """
        # Format the user prompt template with the text to check
        formatted_user_prompt = self.user_prompt.replace("{text}", text_to_check)
        
        # Call CodeAgent if wiki_agentic_rag is enabled else call the LLM model
        llm_response = self._call_llm_with_agentic_rag(formatted_user_prompt) if self.wiki_agentic_rag and self.agent else self._call_llm(formatted_user_prompt)
        
        # Parse and return the response
        return self._parse_response(llm_response)
    
    def _call_llm(self, formatted_user_prompt):
        """
        Make the actual LLM API call with system and user prompts.
        
        Args:
            formatted_user_prompt (str): The formatted user prompt
            
        Returns:
            str: The LLM response or None if the API call fails
        """
        try:
            client = InferenceClient()
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": formatted_user_prompt
                }]

            logger.info(f"Calling LLM model: {self.model} with messages: {messages}")
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            logger.info(f"LLM response: {completion.choices[0].message.content}")
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in _call_llm: {e}")
            return None
    
    def _call_llm_with_agentic_rag(self, formatted_user_prompt):
        """
        Use agentic RAG via CodeAgent.
        The agent will use Wikipedia search tool if needed.
        
        Args:
            formatted_user_prompt (str): The formatted user prompt
            
        Returns:
            str: The agent response or None if the agent run fails
        """
        try:
            prompt = f"{self.system_prompt}\n\n{formatted_user_prompt}"
            logger.info(f"Calling CodeAgent with prompt: {prompt}")        
            response = self.agent.run(prompt)
            logger.info(f"CodeAgent response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error in _call_llm_with_agentic_rag: {e}")
            return None
    
    def _parse_response(self, llm_response):
        """
        Parse and process the LLM response to extract dict with keys 'is_fact_true' and 'reasoning'.
        
        Args:
            llm_response: The raw response from the LLM or CodeAgent
            
        Returns:
            dict or str: Dictionary containing 'is_fact_true' and 'reasoning' keys if parsing succeeds,
                        or None if no suitable result is obtained
        """
        # First try to parse assuming <response> tags are present
        try:
            match = re.search(r'<response>(.*?)</response>', llm_response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                json_data = json.loads(json_str)
                return {
                    "is_fact_true": json_data.get("is_fact_true"),
                    "reasoning": json_data.get("reasoning")
                }
        except Exception as e:
            logger.warning(f"Failed to parse with <response> tags: {e}")
        
        # Fallback: try to parse the response directly as JSON assuming no <response> tags are present
        try:
            json_data = json.loads(llm_response.strip())
            return {
                "is_fact_true": json_data.get("is_fact_true"),
                "reasoning": json_data.get("reasoning")
            }
        except Exception as e:
            logger.error(f"Error parsing response: {e}. Returning raw response: {llm_response}")
            return None