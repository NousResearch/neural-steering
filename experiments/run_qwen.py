#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
from experiments.layer_localization import run_experiment

run_experiment("Qwen/Qwen2.5-3B-Instruct", "experiments/results", top_k=200)
