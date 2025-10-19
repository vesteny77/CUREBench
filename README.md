# Final Phase Result Submission Now Open!

- The **final round dataset** is now available! Please download **`curebench_testset_phase2.jsonl`** from the *Data* section on Kaggle.  
- All data in this new release is **private**, meaning no results will be shown before the deadline.  
- You may still use the **first-round test set** for evaluation to receive immediate feedback and continue refining your model.  
- After the competition concludes, we will **collect valid submissions** for `curebench_testset_phase2.jsonl` and perform **offline evaluation** to determine the final rankings. Evaluation metrics will be announced along with the final results.  
- Please ensure that you **submit your results for `curebench_testset_phase2.jsonl` before the final deadline**, and that your **`meta_file`** includes all required information as outlined in our [GitHub repository](https://github.com/mims-harvard/CUREBench).

# CURE-Bench Starter Kit

[![ProjectPage](https://img.shields.io/badge/CUREBench-Page-red)](https://curebench.ai) [![ProjectPage](https://img.shields.io/badge/CUREBench-Kaggle-green)](https://www.kaggle.com/competitions/cure-bench) [![Q&A](https://img.shields.io/badge/Question-Answer-blue)](QA.md)

A simple inference framework for the CURE-Bench bio-medical AI competition. This starter kit provides an easy-to-use interface for generating submission data in CSV format.

## Updates
 2025.08.08: **[Question&Answer page](QA.md)**: We have created a Q&A page to share all our responses to questions from participants, ensuring fair competition.
 
 2025.09.10: Added starterkit code and tutorials for running **GPT-OSS-20B**, OpenAI’s 20B open-weight reasoning model.

## Quick Start

### Installation Dependencies
```bash
pip install -r requirements.txt
```

## Baseline Setup

If you want to use the ChatGPT baseline:
1. Set up your Azure OpenAI resource
2. Configure environment variables:
```bash
export AZURE_OPENAI_API_KEY_O1="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

If you want to use the open-ended models, (e.g., Qwen, GPT-OSS-20B):
For local models, ensure you have sufficient GPU memory:
```bash
# Install CUDA-compatible PyTorch if needed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transfomers
```

## 📁 Project Structure

```
├── eval_framework.py      # Main evaluation framework
├── dataset_utils.py       # Dataset loading utilities
├── run.py                 # Command-line evaluation script
├── metadata_config.json   # Example metadata configuration
├── requirements.txt       # Python dependencies
└── competition_results/   # Output directory for your results
```

## Dataset Preparation

Download the val and test dataset from the Kaggle site:
```
https://www.kaggle.com/competitions/cure-bench
```

For val set, configure datasets in your `metadata_config_val.json` file with the following structure:
```json
{
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/your/curebench_valset.jsonl",
    "description": "CureBench 2025 val questions"
  }
}
```

For test set, configure datasets in your `metadata_config_test.json` file with the following structure:
```json
{
  "dataset": {
    "dataset_name": "cure_bench_pharse_1",
    "dataset_path": "/path/to/your/curebench_testset.jsonl",
    "description": "CureBench 2025 test questions"
  }
}
```

## Usage Examples

### Basic Evaluation with Config File
```bash
# Run with configuration file (recommended)
python run.py --config metadata_config_test.json
```

## 🔧 Configuration

### Metadata Configuration
Create a `metadata_config_val.json` file. Below is an example:

```json
{
  "metadata": {
    "model_name": "gpt-4o-1120",
    "model_type": "ChatGPTModel",
    "track": "internal_reasoning",
    "base_model_type": "API",
    "base_model_name": "gpt-4o-1120",
    "dataset": "cure_bench_phase_1",
    "additional_info": "Zero-shot ChatGPT run",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  },
  "dataset": {
    "dataset_name": "cure_bench_phase_1",
    "dataset_path": "/path/to/curebench_valset.jsonl",
    "description": "CureBench 2025 val questions"
  },
  "output_dir": "competition_results",
  "output_file": "submission.csv"
}
```

**Notes:**

* Other API models and open-weight models (e.g. Qwen) can be used in the same way
* For fine-tuned model (e.g. GPT-OSS-20B) replace `"model_name"` with your fine-tuned checkpoint, e.g.:
```json
"model_name": "myuser/gpt-oss-20b-curebench-ft"
```

### Required Metadata Fields
- `model_name`: Display name of your model
- `track`: Either "internal_reasoning" or "agentic_reasoning"
- `base_model_type`: Either "API" or "OpenWeighted"
- `base_model_name`: Name of the underlying model
- `dataset`: Name of the dataset

Note: You can leave the following fields empty for the first round of submissions:
`additional_info`,`average_tokens_per_question`, `average_tools_per_question`, and `tool_category_coverage`.
**Please ensure these fields are filled for the final submission.**


### Question Type Support
The framework handles three distinct question types:
1. **Multiple Choice**: Questions with lettered options (A, B, C, D, E)
2. **Open-ended Multiple Choice**: Open-ended questions converted to multiple choice format  
3. **Open-ended**: Free-form text answers


## Output Format

The framework generates submission files in CSV format with a zip package containing metadata. The CSV structure includes:
- `id`: Question identifier
- `prediction`: Model's answer (choice for multiple choice, text for open-ended)
- `reasoning_trace`: Model's reasoning process
- `choice`: The choice for the multi-choice questions.

The metadata structure (example):
```json
{
  "meta_data": {
    "model_name": "gpt-4o-1120",
    "track": "internal_reasoning",
    "model_type": "ChatGPTModel",
    "base_model_type": "API", 
    "base_model_name": "gpt-4o-1120",
    "dataset": "cure_bench_pharse_1",
    "additional_info": "Zero-shot ChatGPT run",
    "average_tokens_per_question": "",
    "average_tools_per_question": "",
    "tool_category_coverage": ""
  }
}
```

## 🤖 Agentic Pipeline (Track 2)

The starter kit now ships with a configurable multi-agent workflow for Track&nbsp;2 (`agentic_reasoning`). Key capabilities:

- **Role-based orchestration** – Planner → Tool router → Evidence workers → Verifier → Decider, with optional self-consistency voting.
- **Model mixing** – Assign GPT‑5 (Azure OpenAI) and Gemini 2.5 Pro to different roles via `runtime.agent_roles`.
- **Tool hooks** – Optional connectors for TxAgent and ToolUniverse. Configure under `runtime.tools` with `enabled`, `providers`, and endpoints.
- **Telemetry auto-fill** – `average_tools_per_question`, `average_tokens_per_question`, and `tool_category_coverage` are written to `meta_data.json` automatically when telemetry is available.

### Quick configs

- Single model (GPT‑5): `config-examples/metadata_config_val_gpt5.json`
- Single model (Gemini 2.5 Pro): `config-examples/metadata_config_val_gemini.json`
- Hybrid (GPT‑5 planner & verifier + Gemini workers & decider): `config-examples/metadata_config_val_hybrid.json`

Run on the validation set (subset of 100 examples shown):

```bash
python run.py --config CUREBench/config-examples/metadata_config_val_hybrid.json --subset-size 100
```

To enable real tool calls, set `runtime.tools.enabled` to `true` and provide endpoints/keys, e.g.:

```json
{
  "runtime": {
    "model_type": "multiagent",
    "tools": {
      "enabled": true,
      "providers": ["txagent", "tooluniverse"],
      "txagent_endpoint": "https://<txagent-host>",
      "txagent_api_key": "...",
      "tooluniverse_endpoint": "https://<tu-host>",
      "timeout": 25
    }
  }
}
```

### Environment variables

| Purpose | Variables |
|---------|-----------|
| Azure OpenAI (GPT‑5) | `AZURE_OPENAI_API_KEY_O1` / `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT` / `AZURE_ENDPOINT`, optional `AZURE_API_VERSION` |
| Google Gemini | `GOOGLE_API_KEY` (or `GOOGLE_APPLICATION_CREDENTIAL` pointing to JSON with `api_key`) |
| TxAgent (optional) | `TXAGENT_ENDPOINT`, `TXAGENT_API_KEY` |
| ToolUniverse (optional) | `TOOLUNIVERSE_ENDPOINT` |

## Model Tutorials

* Step-by-step tutorial for running [OpenAI’s open-weight 20B model](https://huggingface.co/openai/gpt-oss-20b) on CUREBench: [tutorials/gpt-oss-20b/tutorial_gptoss20b.md](tutorials/gpt-oss-20b/tutorial_gptoss20b.md)

## Support

For issues and questions: 
1. Check the error messages (they're usually helpful!)
2. Ensure all dependencies are installed
3. Review the examples in this README
4. Open an Github Issue.

Happy competing!
