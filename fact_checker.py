import os
import yaml
import json
import re
import logging
from huggingface_hub import InferenceClient
from smolagents import CodeAgent, InferenceClientModel, WikipediaSearchTool
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class FactChecker:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the FactChecker from a config file.
        
        Args:
            config_path (str): Path to the config YAML file. Defaults to "config.yaml".
        """
        config = self._load_config(config_path)
        self.wiki_agentic_rag = config.get("wiki_agentic_rag", False)
        self.deployment_type = config.get("deployment_type", "inference_client")
        # If wiki_agentic_rag is enabled, deployment_type must be "inference_client" 
        if self.wiki_agentic_rag and self.deployment_type != "inference_client":
            logger.warning(f"wiki_agentic_rag is enabled but deployment_type is '{self.deployment_type}'. Forcing deployment_type to 'inference_client'.")
            self.deployment_type = "inference_client"
        self.model = config.get("model", "openai/gpt-oss-120b")
        self.temperature = config.get("temperature", 0)
        self.system_prompt = self._load_prompt_template(config.get("system_prompt_path", "prompts/system_prompt.txt"))
        self.user_prompt = self._load_prompt_template(config.get("user_prompt_path", "prompts/user_prompt.txt"))

        # Initialize model and tokenizer if local deployment type.
        self.loaded_tokenizer = None
        self.loaded_model = None
        if self.deployment_type == "local":
            logger.info(f"Loading local model: {self.model} (this may take a moment...)")
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.loaded_model = AutoModelForCausalLM.from_pretrained(self.model)

        #Initialize InferenceClient if inference client deployment type and not wiki_agentic_rag
        self.client = None
        if not self.wiki_agentic_rag and self.deployment_type == "inference_client":
            self.client = InferenceClient(self.model)
        
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
            #Fallback to default prompt
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

    def _call_llm_hf_inference_client(self, messages):
        """
        Make the actual LLM API call with system and user prompts using Hugging Face InferenceClient.
        
        Args:
            messages (list): The list of messages to be sent to the LLM
            
        Returns:
            str: The LLM response or None if the API call fails
        """
        logger.info(f"Calling HF InferenceClient LLM model: {self.model} with messages: {messages}")
        completion = self.client.chat.completions.create(
            messages=messages,
            temperature=self.temperature,
        )
        response = completion.choices[0].message.content
        logger.info(f"HF InferenceClient LLM response: {response}")
        return response

    def _call_llm_hf_locally_hosted_model(self, messages):
        """
        Make the actual LLM API call with system and user prompts using locally hosted model.
        
        Args:
            messages (list): The list of messages to be sent to the LLM
            
        Returns:
            str: The LLM response or None if the API call fails
        """
        logger.info(f"Calling locally hosted LLM model: {self.model} with messages: {messages}")
        inputs = self.loaded_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.loaded_model.device)
        outputs = self.loaded_model.generate(**inputs, max_new_tokens=512)
        response = self.loaded_tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])
        logger.info(f"LLM response: {response}")
        return response
        
    def _call_llm(self, formatted_user_prompt):
        """
        Make the actual LLM API call with system and user prompts.
        Calls either inference client or locally hosted model based on deployment_type.
        
        Args:
            formatted_user_prompt (str): The formatted user prompt
            
        Returns:
            str: The LLM response or None if the API call fails
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": formatted_user_prompt
                }]

            # Route to appropriate method based on deployment type
            if self.deployment_type == "local":
                return self._call_llm_hf_locally_hosted_model(messages)
            else:
                return self._call_llm_hf_inference_client(messages)
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
            json_str = match.group(1).strip()
            json_data = json.loads(json_str)
            return {
                "is_fact_true": json_data.get("is_fact_true"),
                "reasoning": json_data.get("reasoning")
            }
        except Exception as e:
            logger.warning(f"Failed to parse with <response> tags: {e}")
        
        # Fallback: try to parse the response assuming no <response> tags are present
        try:
            if isinstance(llm_response, str):
                json_data = json.loads(llm_response.strip())
                return {
                    "is_fact_true": json_data.get("is_fact_true"),
                    "reasoning": json_data.get("reasoning")
                }
            elif isinstance(llm_response, dict):
                return {
                    "is_fact_true": llm_response.get("is_fact_true"),
                    "reasoning": llm_response.get("reasoning")
                }
            else:
                logger.error(f"Unsupported response type: {type(llm_response)}")
        except Exception as e:
            logger.error(f"Error parsing response: {e}. Returning raw response: {llm_response}")
        
        return None