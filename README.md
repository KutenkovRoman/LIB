# 🚀 Optimization Library (LIB)

This library provides a comprehensive framework for experimenting with various optimization algorithms across different machine learning tasks. The library supports multiple datasets and models, with a special focus on optimization strategies.

## 🛠️ Setup

### Environment Setup

You can create the required environment using the provided `environment.yaml` file:

```bash
conda env create -f environment.yaml
conda activate optim_lib
```

Alternatively (to avoid setting up conda) you could use `requirements.txt` file from main branch:
```bash
python -m venv optim_venv
source optim_venv/bin/activate
pip install -r requirements.txt
```

## 📁 Project Structure

The project is organized into several key directories:

- `src/` - Core source code
  - `config.py` - Main configuration parser
  - `libsvm/` - LIBSVM datasets and models
  - `cv/` - Computer Vision datasets and models
  - `fine_tuning/` - Fine-tuning strategies for pre-trained models
  - `optimizers/` - Implementation of various optimization algorithms
- `scripts/` - Ready-to-use scripts for running experiments
- `data/` - Default location for datasets
- `notebooks/` - Example notebooks

## ⚙️ Argument System

The library uses a hierarchical argument system:

1. **Base Arguments** (`config.py`): Core arguments applicable to all experiments
2. **Task-Specific Arguments**: Extended arguments for specific tasks, specifically
   - Fine-Tuning Arguments (`fine_tuning/config_ft.py`)

Arguments are processed hierarchically. When running an experiment:
1. Base arguments are loaded first
2. Based on the selected dataset, task-specific arguments are added
3. If a configuration file is specified with `--config_name`, its values override defaults

## 🔧 How to run code

There are example scripts `Llama2_alt.sh` and `Qwen.sh` located in `./scripts/style` that have basic setup. You must set `dataset=style` and specify `--dataset_path` to choose dataset (stored locally). To use `wandb` (if available) set flag `--wandb` and specify `--wandb_project`. In order to save adapters after fine-tuning set `save_strategy=steps/epoches` and specify `--save_name`, results will be stored in `./scr/fine_tuning/style/results_raw/{save_name}`. Some models (e.g. `Llama-2-7b-hf`) require hf-token to load.
