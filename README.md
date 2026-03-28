# FVRuleGen

FVRuleGen is a framework for automatically generating SystemVerilog Assertions (SVA) from Natural Language (NL) instructions and RTL designs, and evaluating the generated assertions using LLM-based agents.

## Installation

1. Create a conda environment using the YAML file:
   ```bash
   conda env create -f environment.yaml
   conda activate fvrulegen
   ```

2. (Optional) If you are in an internal environment, configure the global index url:
   ```bash
   pip config set global.index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple
   # Then reinstall adlrchat to ensure it comes from the internal source if needed
   pip install -U adlrchat
   ```

## Usage

The project is primarily run through the Hardware Agent using `main.py` with configuration specified in `config.py`.

### 1. Configuration

Edit the configuration file at `hardware-agent-marco/src/config.py` to set up your experiment parameters.

Key configurations to modify:
- **Global Task**: Set `global_task` to `'inference'`, `'train'`, or `'eval'`. Inference is LLM+Evaluation, and eval is evaluation itself.
- **Task Type**: Set `task` to your target task (e.g., `'nl2sva_human'`, `'nl2sva_machine'`, `'nl2sva_opencore'`, `'design2sva_...'`).
- **LLM Model**: Configure `llm_model` (e.g., `'o3-mini-20250131'`, `'gpt-4o'`, `'claude-sonnet-4-5-...`').
- **Debug Mode**: Set `debug = True` for testing with a smaller subset of data.

```python
# Example config.py settings
global_task = 'inference'
task = "nl2sva_opencore"
llm_model = 'o3-mini-20250131'
debug = True
```

### 2. Execution

Run the agent from the `hardware-agent-marco/src` directory:

```bash
cd hardware-agent-marco/src
python main.py
```

### 3. Evaluation

Evaluation is handled within the framework based on the `global_task` setting in `config.py`. 
- If `global_task` is `'eval'`, the script will run evaluation logic.
- Ensure `use_JG = True` in `config.py` to enable JasperGold evaluation where applicable.

## Directory Structure

- `hardware-agent-marco/src/`: Core source code for the agent.
  - `main.py`: Entry point for the agent.
  - `config.py`: Main configuration file.
  - `fv_harwareagent_example.py`: Main processor class (`FVProcessor`).
- `data_svagen/`: Datasets.
- `fv_eval/`: Core evaluation package code.

## Licenses

Copyright © 2026, NVIDIA Corporation. All rights reserved.
This work is made available under the NVIDIA License.
