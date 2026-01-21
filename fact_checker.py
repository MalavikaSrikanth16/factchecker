import json
import logging
import re
import yaml
from typing import Dict, Any, Optional, List, Union
from huggingface_hub import InferenceClient
from smolagents import CodeAgent, InferenceClientModel, WikipediaSearchTool
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import *
import torch
from IzzyViz.izzyviz import visualize_attention_self_attention

logger = logging.getLogger(__name__)

class FactChecker:
    def __init__(self, config_path: str = CONFIG_PATH) -> None:
        """
        Initialize the FactChecker from a config file.
        
        Args:
            config_path (str): Path to the config YAML file. Defaults to "config.yaml".
        """
        config = FactChecker.load_config(config_path)
        self.wiki_agentic_rag = config.get(CONFIG_KEY_WIKI_AGENTIC_RAG, False)
        self.deployment_type = config.get(CONFIG_KEY_DEPLOYMENT_TYPE, CONFIG_INFERENCE_CLIENT_DEPLOYMENT_TYPE)
        # If wiki_agentic_rag is enabled, deployment_type must be "inference_client" 
        if self.wiki_agentic_rag and self.deployment_type != CONFIG_INFERENCE_CLIENT_DEPLOYMENT_TYPE:
            logger.warning(f"wiki_agentic_rag is enabled but deployment_type is '{self.deployment_type}'. Forcing deployment_type to 'inference_client'.")
            self.deployment_type = CONFIG_INFERENCE_CLIENT_DEPLOYMENT_TYPE
        self.model = config.get(CONFIG_KEY_MODEL, FALLBACK_MODEL)
        self.temperature = config.get(CONFIG_KEY_TEMPERATURE, 0)
        self.system_prompt = FactChecker.load_prompt_template(
            config.get(CONFIG_KEY_SYSTEM_PROMPT_PATH, FALLBACK_SYSTEM_PROMPT_PATH)
        )
        self.user_prompt = FactChecker.load_prompt_template(
            config.get(CONFIG_KEY_USER_PROMPT_PATH, FALLBACK_USER_PROMPT_PATH)
        )
        self.izzyviz = config.get(CONFIG_KEY_IZZYVIZ, False)

        self._initialize_model_components()

    def _initialize_model_components(self) -> None:
        """Initialize model, client, and agent components based on deployment type."""
        # Initialize model and tokenizer if local deployment type
        self.loaded_tokenizer = None
        self.loaded_model = None
        if self.deployment_type == CONFIG_LOCAL_DEPLOYMENT_TYPE:
            logger.info(f"Loading local model: {self.model} (this may take a moment...)")
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.loaded_model = AutoModelForCausalLM.from_pretrained(self.model)

        # Initialize InferenceClient if inference client deployment type and not wiki_agentic_rag
        self.client = None
        if not self.wiki_agentic_rag and self.deployment_type == CONFIG_INFERENCE_CLIENT_DEPLOYMENT_TYPE:
            logger.info(f"Initializing InferenceClient with model: {self.model}")
            self.client = InferenceClient(self.model)

        # Initialize agent with Wikipedia search tool if wiki_agentic_rag is enabled
        self.agent = None
        if self.wiki_agentic_rag:
            logger.info(f"Initializing CodeAgent with Wikipedia search tool")
            #TODO : These options should be read from config
            wiki_tool = WikipediaSearchTool(
                user_agent="FactChecker/1.0 (fact-checking-tool)",
                language="en",
                content_type="text",
                extract_format="WIKI"
            )
            agent_model = InferenceClientModel(
                model_id=self.model,
                temperature=self.temperature
            )
            self.agent = CodeAgent(tools=[wiki_tool], model=agent_model)

    @staticmethod
    def load_prompt_template(prompt_path: str) -> str:
        """
        Load a prompt template from a file.
        
        Args:
            prompt_path (str): Path to the prompt template file.
            
        Returns:
            str: The prompt template string (default prompt string is returned if file not found).
        """
        try:
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            # Fallback to default prompt
            logger.warning(f"Error loading prompt template from {prompt_path}, falling back to default prompt: {e}")
            return "Fact check the following text: {text}. Return JSON output with keys 'is_fact_true' and 'reasoning'"
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
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
    
    def check_fact(self, text_to_check: str) -> Optional[Dict[str, Any]]:
        """
        Fact-check the given text using the LLM.
        If wiki_agentic_rag is true, uses agentic RAG with Wikipedia search tool.
        
        Args:
            text_to_check (str): The text that needs to be fact-checked
            
        Returns:
            dict or str: Dictionary containing 'is_fact_true' and 'reasoning' keys,
                        or None if no suitable result is obtained
        """
        # Format the user prompt
        formatted_user_prompt = self._format_user_prompt(text_to_check)

        # Call LLM without RAG or LLM with Agentic RAG based on whether agentic_rag is enabled
        agentic_rag_enabled = self.wiki_agentic_rag and self.agent
        llm_response = self._call_llm_with_agentic_rag(formatted_user_prompt) if agentic_rag_enabled else self._call_llm(formatted_user_prompt)
        
        # Parse and return the response
        return self._parse_response(llm_response)

    def _format_user_prompt(self, text_to_check: str) -> str:
        """
        Format the user prompt with the text to check.
        
        Args:
            text_to_check (str): The text that needs to be fact-checked
            
        Returns:
            str: The formatted user prompt
        """
        return self.user_prompt.replace(USER_PROMPT_TEXT_PLACEHOLDER, text_to_check)

    def _generate_self_attention_heatmap(self, text_to_visualize: str) -> None:
        """
        Generates a self-attention heatmap for the given text using IzzyViz.
        
        Args:
            text_to_visualize (str): The text to visualize
        """
        try:
            inputs = self.loaded_tokenizer(text_to_visualize, return_tensors="pt", add_special_tokens=True)
            tokens = self.loaded_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            self.loaded_model.set_attn_implementation(ATTENTION_IMPLEMENTATION_EAGER)
            with torch.no_grad():
                outputs = self.loaded_model(**inputs, output_attentions=True)
                attentions = outputs.attentions   
            #TODO : These options should be read from config
            visualize_attention_self_attention(
                attentions=attentions,
                tokens=tokens,
                layer=-1,
                head=0,
                top_n=5,
                mode='self_attention',
                plot_titles=["Fact Checking Self-Attention Heatmap"],
                save_path="attention_heat_maps/fact_check_attention_heatmap.png"
            )
            logger.info("Self-attention heatmap saved to attention_heat_maps/fact_check_attention_heatmap.png")
        except Exception as e:
            logger.error(f"Error generating self-attention heatmap: {e}")

    def _call_llm_hf_inference_client(self, messages):
        """
        Calls the LLM using Hugging Face InferenceClient.
        
        Args:
            messages (list): The list of messages to be sent to the LLM
            
        Returns:
            str: The LLM response or None if there is an error.
        """
        logger.info(f"Calling HF InferenceClient LLM: {self.model} with messages: {messages}")
        completion = self.client.chat.completions.create(
            messages=messages,
            temperature=self.temperature,
        )
        response = completion.choices[0].message.content
        logger.info(f"LLM response: {response}")
        return response

    def _call_llm_hf_locally_hosted_model(self, messages):
        """
        Calls the locally hosted LLM model.
        
        Args:
            messages (list): The list of messages to be sent to the LLM
            
        Returns:
            str: The LLM response or None if there is an error.
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
        response = self.loaded_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:]
        )
        logger.info(f"LLM response: {response}")
        if self.izzyviz:
            input_and_response = self.loaded_tokenizer.decode(outputs[0])
            self._generate_self_attention_heatmap(input_and_response)

        return response

    def _call_llm(self, formatted_user_prompt):
        """
        Make the LLM model call with system and user prompts.
        Calls either HF inference client or locally hosted model based on deployment_type.
        
        Args:
            formatted_user_prompt (str): The formatted user prompt
            
        Returns:
            str: The LLM response or None if there is an error
        """
        try:
            # Create messages list for the LLM
            messages = [
                {
                    "role": SYSTEM_ROLE,
                    "content": self.system_prompt
                },
                {
                    "role": USER_ROLE,
                    "content": formatted_user_prompt
                }]
            # Route to appropriate method based on deployment type
            if self.deployment_type == CONFIG_LOCAL_DEPLOYMENT_TYPE:
                return self._call_llm_hf_locally_hosted_model(messages)
            else:
                return self._call_llm_hf_inference_client(messages)
        except Exception as e:
            logger.error(f"Error in _call_llm: {e}")
            return None
    
    def _call_llm_with_agentic_rag(self, formatted_user_prompt: str) -> Optional[str]:
        """
        Calls the LLM with agentic RAG using system and user prompts.
        The agent will use Wikipedia search tool if needed.
        
        Args:
            formatted_user_prompt (str): The formatted user prompt
            
        Returns:
            str: The agent response or None if the agent run fails
        """
        try:
            llm_with_agentic_rag_prompt = f"{self.system_prompt}\n\n{formatted_user_prompt}"
            logger.info(f"Calling LLM with agentic RAG with prompt: {llm_with_agentic_rag_prompt}")
            response = self.agent.run(llm_with_agentic_rag_prompt)
            logger.info(f"LLM with agentic RAG response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error in _call_llm_with_agentic_rag: {e}")
            return None

    def _parse_response(self, llm_response: Union[str, Dict[str, Any], None]) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response to extract dict with keys 'is_fact_true' and 'reasoning'.
        
        Args:
            llm_response: The response from the LLM (With or without Agentic RAG)

        Returns:
            dict or None: Dictionary containing 'is_fact_true' and 'reasoning' keys if parsing succeeds,
                         or None if no suitable result is obtained
        """
        # First try to parse assuming <response> tags are present
        try:
            match = re.search(r'<response>(.*?)</response>', llm_response, re.DOTALL)
            json_str = match.group(1).strip()
            json_data = json.loads(json_str)
            return {
                "is_fact_true": json_data.get(LLM_RESPONSE_KEY_IS_FACT_TRUE),
                "reasoning": json_data.get(LLM_RESPONSE_KEY_REASONING)
            }
        except Exception as e:
            logger.warning(f"Failed to parse with <response> tags, trying without tags: {e}")

        # Fallback: try to parse the response assuming no <response> tags are present
        try:
            if isinstance(llm_response, str):
                json_data = json.loads(llm_response.strip())
                return {
                    "is_fact_true": json_data.get(LLM_RESPONSE_KEY_IS_FACT_TRUE),
                    "reasoning": json_data.get(LLM_RESPONSE_KEY_REASONING)
                }   
            elif isinstance(llm_response, dict):
                return {
                    "is_fact_true": llm_response.get(LLM_RESPONSE_KEY_IS_FACT_TRUE),
                    "reasoning": llm_response.get(LLM_RESPONSE_KEY_REASONING)
                }
            else:
                logger.error(f"Error parsing LLM response: Unsupported response type: {type(llm_response)}")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")

        return None