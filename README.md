# Fact Checker

A Streamlit-based fact-checking application that uses Large Language Models (LLMs) to verify claims and provide reasoning. The application supports both locally hosted models and using the Hugging Face Inference API, with optional agentic RAG capabilities using Wikipedia search.

## Features

- **Multiple Deployment Options**: Use locally hosted models or Hugging Face Inference API
- **Agentic RAG**: Optional agentic RAG with wikipedia search integration 
- **Configurable Prompts**: Customize system and user prompts via text files
- **Interactive UI**: Clean Streamlit interface for easy fact-checking

## Installation and setup

1. Clone the repository:
```bash
git clone <repository-url>
cd factchecker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up authentication (if using Hugging Face Inference API):
   - Create a `.streamlit/secrets.toml` file (if it doesn't exist)
   - Add your Hugging Face token to the file:
    ```toml
    HF_TOKEN = "your_huggingface_token_here"
    ```
    - Alternatively, you can set the `HF_TOKEN` environment variable.
    - Get your token from [Hugging Face Access Tokens](https://huggingface.co/settings/tokens).

## Setting the Configuration

The application is configured via `config.yaml`. Here's a detailed explanation of each option:

### deployment_type

**Type**: `string`  
**Options**: `"local"` | `"inference_client"`  
**Default**: `"inference_client"`

Determines how the LLM is accessed:

- `"inference_client"`: Uses Hugging Face Inference API. Requires a Hugging Face token. Suitable for larger models that can't run locally.
- `"local"`: Uses a locally hosted model loaded via Transformers. No Hugging Face token required. Suitable for smaller models.

### model

**Type**: `string`  
**Default**: `"openai/gpt-oss-120b"`

The model identifier to use for fact-checking:

- For `inference_client`: Use models available on Hugging Face Inference API. Examples:
  - `"openai/gpt-oss-120b"` (tested, large model)
  - Other models available via HF Inference API
  
- For `local`: Use smaller models that can be loaded locally. Examples:
  - `"HuggingFaceTB/SmolLM2-1.7B-Instruct"` (tested, small model)
  - Other models compatible with Transformers library

  **Note**: `"HuggingFaceTB/SmolLM2-1.7B-Instruct"` is NOT available via HF Inference API.

### system_prompt_path

**Type**: `string`  
**Default**: `"prompts/system_prompt.txt"`

Path to the system prompt template file. This prompt defines the role and behavior of the LLM for fact-checking tasks. 

### user_prompt_path

**Type**: `string`  
**Default**: `"prompts/user_prompt.txt"`

Path to the user prompt template file. This prompt is used to format the user's input text for fact-checking. The template should include a `{text}` placeholder where the user's claim will be inserted.

**Example user prompt template**:
```
Fact check the following text: {text}. Return JSON output with keys 'is_fact_true' and 'reasoning'
```

### temperature

**Type**: `float`  
**Default**: `0`

Controls the randomness of the model's responses:
- `0`: Deterministic, focused responses (recommended for fact-checking)
- `Higher values` : More creative/varied responses

For fact-checking, lower values are typically preferred for consistency.

### wiki_agentic_rag

**Type**: `boolean`  
**Default**: `true`

Enables agentic RAG (Retrieval-Augmented Generation) using Wikipedia search:

- `true`: The fact-checker uses a CodeAgent with Wikipedia search tool to look up information before making fact-checking decisions.
- `false`: Agentic RAG is disabled.

**Note**: If `wiki_agentic_rag` is enabled, `deployment_type` must be `"inference_client"` currently and Hugging Face Authentication token is required.

## Example Configuration

```yaml
# Use Hugging Face Inference API with agentic RAG
deployment_type: "inference_client"
model: "openai/gpt-oss-120b"
system_prompt_path: "prompts/system_prompt.txt"
user_prompt_path: "prompts/user_prompt.txt"
temperature: 0
wiki_agentic_rag: true
```

```yaml
# Use local model 
deployment_type: "local"
model: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
system_prompt_path: "prompts/system_prompt.txt"
user_prompt_path: "prompts/user_prompt.txt"
temperature: 0
wiki_agentic_rag: false
```

## Running the Application

Start the Streamlit interface:

```bash
streamlit run chat_interface.py
```

The application will open in your default web browser. Enter a statement in the text area and click "Check Fact" to get a fact check result with reasoning.

## Project Structure

```
WINGProject/
├── config.yaml              # Configuration file
├── chat_interface.py        # Streamlit UI file
├── fact_checker.py          # Fact-checking logic
├── requirements.txt         # Python dependencies
├── prompts/
│   ├── system_prompt.txt    # System prompt template
│   └── user_prompt.txt      # User prompt template
└── .streamlit/
    └── secrets.toml         # Secrets (HF_TOKEN)
```

## Dependencies

- `streamlit`: Web application framework
- `pyyaml`: YAML configuration parsing
- `huggingface-hub`: Hugging Face API client
- `smolagents`: Agentic RAG framework
- `wikipedia-api`: Wikipedia search integration
- `transformers`: Local model loading


## Troubleshooting

- **"HF_TOKEN is missing"**: Ensure your Hugging Face token is set in `.streamlit/secrets.toml` or as an environment variable if using HF Inference API.
- **Model loading errors**: For local models, ensure you have sufficient disk space and memory. Large models may require significant resources.