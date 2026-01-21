# Fact Checker

A Streamlit-based fact-checking application that uses Large Language Models (LLMs) to fact-check texts and provide reasoning. The application supports both locally hosted LLMs and using LLMs via the Hugging Face Inference API. The application supports optional agentic RAG capabilities using Wikipedia search.

## Features

- **Multiple LLM Access Options**: Use locally hosted LLMs or LLMs via Hugging Face Inference API
- **Agentic RAG**: Optional agentic RAG with wikipedia search integration 
- **Configurable Prompts**: Customize system and user prompts via text files
- **Interactive UI**: Clean Streamlit interface for easy fact-checking

## Installation and setup

1. Clone the repository:
```bash
git clone https://github.com/MalavikaSrikanth16/factchecker.git
cd factchecker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. <a id="huggingface-auth"></a>[Required if using LLMs via Hugging Face Inference API or Agentic RAG] Set up huggingface authentication :
   - Get your Hugging Face token from [Hugging Face Access Tokens](https://huggingface.co/settings/tokens).
   - Create a `.streamlit/secrets.toml` file (if it doesn't exist)
   - Add your Hugging Face token to the file:
    ```toml
    HF_TOKEN = "your_huggingface_token_here"
    ```
    - Alternatively, you can set the `HF_TOKEN` environment variable.
    
4. <a id="izzyviz-install"></a>[Optional] To use the `izzyviz` feature to visualize LLM self-attention heat maps, you need to install IzzyViz :

- Install from GitHub
```bash
git clone https://github.com/WING-NUS/IzzyViz.git
cd IzzyViz
pip install -e .
```

## Setting the Configuration

The application is configured via `config.yaml`. Here's a detailed explanation of how to set each option:

### deployment_type

**Type**: `string`  
**Value Options**: `"local"` | `"inference_client"`  
**Default Value**: `"inference_client"`

Determines how the LLM is accessed:

- `"inference_client"`: Uses LLMs via Hugging Face Inference API. Suitable for larger models that can't run locally. **[Requires setting up Hugging Face Authentication](#huggingface-auth).** 
- `"local"`: Uses a locally hosted LLM. Suitable for smaller models.

### model

**Type**: `string`  
**Default Value**: `"openai/gpt-oss-120b"`

The model identifier to use for fact-checking:

- For `inference_client` deployment_type : Use LLMs available on Hugging Face Inference API. Example: `"openai/gpt-oss-120b"` 
- For `local`: Use smaller LLMs that can be loaded locally and are compatible with the HuggingFace Transformers library. Example: `"HuggingFaceTB/SmolLM2-1.7B-Instruct"` 

  **Note**: `"HuggingFaceTB/SmolLM2-1.7B-Instruct"` is NOT available via Hugging Face Inference API.

### system_prompt_path

**Type**: `string`  
**Default**: `"prompts/system_prompt.txt"`

Path to the system prompt template file. The system prompt defines the role and behavior of the fact checking LLM.

### user_prompt_path

**Type**: `string`  
**Default**: `"prompts/user_prompt.txt"`

Path to the user prompt template file. The user prompt is used to format the user's input text for fact-checking. The template should include a `{text}` placeholder where the user's input text will be inserted.

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

### wiki_agentic_rag

**Type**: `boolean`  
**Default**: `true`

Enables agentic RAG (Retrieval-Augmented Generation) using Wikipedia search:

- `true`: The fact-checker uses a CodeAgent with Wikipedia search tool to look up and refine information before making fact-checking decisions.
- `false`: Agentic RAG is disabled.

**Note**: If `wiki_agentic_rag` is enabled :
- `deployment_type` must be `"inference_client"` currently
- **[Setting up Hugging Face Authentication is required](#huggingface-auth).** 

### izzyviz

**Type**: `boolean`  
**Default**: `false`

Enables LLM self-attention heatmap visualization using IzzyViz:

- `true`: Generates self-attention heatmaps for the LLM input text and the fact checker response. The heatmap is typically saved to `attention_heat_maps/fact_check_attention_heatmap.png`.
- `false`: Self-attention visualization is disabled.

**Note**: `izzyviz` will only work if:
- `deployment_type` is set to `"local"` (not available with `"inference_client"` deployment_type)
- IzzyViz is installed. [**See installation instructions**](#izzyviz-install)


## Example Configuration

```yaml
# Use LLM via Hugging Face Inference API with agentic RAG 
deployment_type: "inference_client"
model: "openai/gpt-oss-120b"
system_prompt_path: "prompts/system_prompt.txt"
user_prompt_path: "prompts/user_prompt.txt"
temperature: 0
wiki_agentic_rag: true
izzyviz: false
```

```yaml
# Use locally hosted LLM 
deployment_type: "local"
model: "HuggingFaceTB/SmolLM2-1.7B-Instruct"
system_prompt_path: "prompts/system_prompt.txt"
user_prompt_path: "prompts/user_prompt.txt"
temperature: 0
wiki_agentic_rag: false
izzyviz: false
```

## Running the Application

After Setup and Installation and Setting the Configuration, you can start the Streamlit interface:

```bash
streamlit run chat_interface.py
```

The application will open in your default web browser. Enter a statement in the text area and click "Check Fact" to get the fact check result along with reasoning.

## Project Structure

```
factchecker/
├── config.yaml              # Configuration file
├── chat_interface.py        # Streamlit UI file
├── fact_checker.py          # Fact-checking logic
├── constants.py             # Utilized constants 
├── requirements.txt         # Python dependencies
├── prompts/
│   ├── system_prompt.txt    # System prompt template
│   └── user_prompt.txt      # User prompt template
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_fact_checker_unit.py         # Unit tests
│   ├── test_fact_checker_integration.py  # Integration tests
├── attention_heat_maps/     # Generated attention heatmap visualizations
├── .streamlit/
│   └── secrets.toml         # Secrets (HF_TOKEN)
```

## Dependencies

- `streamlit`: Web application framework
- `pyyaml`: YAML configuration parsing
- `huggingface-hub`: Hugging Face API client
- `smolagents`: Agentic RAG framework
- `wikipedia-api`: Wikipedia search integration
- `transformers`: Local model loading
- `torch` : PyTorch framework
- `pytest` : Testing framework

## Testing

The project includes both unit tests and integration tests to ensure the fact-checking functionality works correctly.

### Running Tests

To run all tests:
```bash
pytest tests/
```

To run only unit tests:
```bash
pytest tests/test_fact_checker_unit.py
```

To run only integration tests:
```bash
pytest tests/test_fact_checker_integration.py
```

### Unit Tests

The unit tests in `test_fact_checker_unit.py` test individual components and methods of the `FactChecker` class:

1. **`TestLoadConfig`**: Tests the configuration loading functionality

2. **`TestLoadPromptTemplate`**: Tests the prompt template loading functionality

3. **`TestFactCheckerInitialization`**: Tests the `FactChecker` initialization with different configurations

4. **`TestParseResponse`**: Tests the response parsing functionality

### Integration Tests

The integration tests in `test_fact_checker_integration.py` test the fact-checking flow:

1. **`test_check_fact_with_inference_client`**: Tests the fact-checking flow using the Hugging Face Inference API client. 

2. **`test_check_fact_with_local_model`**: Tests the fact-checking flow using a locally hosted model. 

3. **`test_check_fact_with_agentic_rag`**: Tests the fact-checking flow with agentic RAG enabled. 

4. **`test_check_fact_handles_llm_error`**: Tests error handling when the LLM call fails. 

