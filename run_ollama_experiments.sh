#!/bin/bash


python ollama_experiments.py run experiments/no_codebook_AAC
python ollama_experiments.py run experiments/codebook_AAC
python ollama_experiments.py run experiments/gvfc

python ollama_experiments.py run experiments/icl_experiments/no_codebook/1
python ollama_experiments.py run experiments/icl_experiments/codebook/1

python ollama_experiments.py run experiments/generate_icl_codebook_CoT

python ollama_experiments.py run experiments/cot_experiments/no_codebook/1
python ollama_experiments.py run experiments/cot_experiments/codebook/1

python ollama_experiments.py run experiments/icl_experiments/no_codebook/2
python ollama_experiments.py run experiments/icl_experiments/gemma_ict/codebook/2

python ollama_experiments.py run experiments/cot_experiments/no_codebook/2
python ollama_experiments.py run experiments/cot_experiments/codebook/2
