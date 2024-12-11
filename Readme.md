# Supplementary  Material for 'Just Read the Codebook! Make Use of Quality Codebooks in Zero-Shot Classification of Multilabel Frame Datasets'
These files were used to create the results from the above paper submission to COLING2025. In case of acceptance, all files will be made public.
From this version gvfc are missing due to size constraints. They can be fully be reproduced from the config in the folder.

The code was run on an Ubuntu 22.04 system with an Geforce RTX 3090.
The Nvidia driver version and Cuda drive versions were
`Driver Version: 550.107.02 CUDA Version: 12.4 `.

## Prerequisites
### Installation
You need to install all requirements and [Ollama](https://ollama.com/download) to run experiments.
We reccomend using the Pipfile to install a venv.
### Model pulling
You need to pull the models that we have used from Ollama. 
```bash
ollama pull gemma2:27b
ollama pull llama3.1:8b-instruct-fp16
ollama pull mistral:7b-instruct-v0.3-fp16
```

## Files and folders
All data is found in the folder `datasets`
All experiments are found in the folder `experiments`

`run_all_experiments.sh` is a script that runs all experiments that are needed to reproduce the papers' results.
**Note:** This script will likely run for a few days. You can stop the script and already conducted experiments 
will not be run again. This folder already comes with all experiments conducted and results present, so running 
the script will just skip over all experiments. If you'd like, you can just delete the results 
(anything but the `config.yml` in an experiments folder), and rerun all experiments.

`codebook_experiments.py` contains all python code that is needed for results creation. results can be run from
the command line (examples in the runscript) and need a path to a folder with a config file.

## Config structure
All experiments are run from config files. Check out `example_config.yml` for an in-depth look.

## creating additional stats
If you want to reproduce all extra statistics, you need to process the directories using the function in `ollama_experiments.py`
example code:
```python
from pathlib import Path
import ollama_experiments
Path = 'experiments/gvfc'
ollama_experiments.process_reports('experiments/Codebook/gvfc/')
for file in path.rglob('results.csv'):
    ollama_experiments.additional_stats_from_results(file, no_no_theme=True)
```
For gvfc add flag `no_no_theme=True` and for AAC results add `no_other=True` in order to get statistics for only the relevant 
data points in the datasets.
