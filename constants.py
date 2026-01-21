CONFIG_PATH = "config.yaml"

#Config Keys and Values
CONFIG_KEY_DEPLOYMENT_TYPE = "deployment_type"
CONFIG_INFERENCE_CLIENT_DEPLOYMENT_TYPE= "inference_client"
CONFIG_LOCAL_DEPLOYMENT_TYPE = "local"
CONFIG_KEY_IZZYVIZ = "izzyviz"
CONFIG_KEY_WIKI_AGENTIC_RAG = "wiki_agentic_rag"
CONFIG_KEY_MODEL = "model"
CONFIG_KEY_TEMPERATURE = "temperature"
CONFIG_KEY_SYSTEM_PROMPT_PATH = "system_prompt_path"
CONFIG_KEY_USER_PROMPT_PATH = "user_prompt_path"

#LLM Response Keys
LLM_RESPONSE_KEY_IS_FACT_TRUE = "is_fact_true"
LLM_RESPONSE_KEY_REASONING = "reasoning"

#Fallback Values
FALLBACK_MODEL = "openai/gpt-oss-120b"
FALLBACK_SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
FALLBACK_USER_PROMPT_PATH = "prompts/user_prompt.txt"

#Other Constants
SYSTEM_ROLE = "system"
USER_ROLE = "user"
ATTENTION_IMPLEMENTATION_EAGER = "eager"
USER_PROMPT_TEXT_PLACEHOLDER = "{text}"
HUGGINGFACE_TOKEN = "HF_TOKEN"