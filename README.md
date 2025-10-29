WCST Transformer: Context Switching & Supervision Experiments

This repository implements a Transformer-based model for a synthetic Wisconsin Card Sorting Test (WCST) sequence task.
The goal is to explore how context switching, supervision granularity, and model capacity affect generalization in rule-changing environments.

Required Python Packages:
pip install torch numpy tqdm matplotlib

Reproducibility Notice:
Each run of this model involves stochastic data generation and random initialization.
While results should remain consistent in trend, exact values (e.g., accuracy or loss) may differ slightly between runs.
This is expected behavior due to:
    the random WCST rule generation process, and
    the randomized initialization of model parameters.
We’ve included all original runs under the runs/ folder so that you can inspect the exact configurations and metrics used for our experiments.

Quick Start:
Train and evaluate the main model (Model 3):
python main.py --switch_period 64 --supervise all

Key Arguments:
Argument                        Description                                             Example
--switch_period	                Context switch frequency (none, 32, 64, 128)	        --switch_period 64
--supervise	                    Which SEP positions to supervise (all, last, query)	    --supervise all
--train_batches	                Number of training batches	                            --train_batches 2000
--val_batches, --test_batches	Validation/test batch counts	                        --val_batches 300
--d_model	                    Model hidden size	                                    --d_model 128
--num_layers, --num_heads	    Transformer depth/width	                                --num_layers 4 --num_heads 4
--epochs	                    Training epochs	                                        --epochs 10

After training, results are automatically saved under:
runs/<timestamp>_<config_tag>/
Each run contains:
config.json   : full training configuration
metrics.json  : final test loss/accuracy/confusion
history.json  : epoch-wise metrics
confusion.png : normalized confusion heatmap
model.pt      : best checkpoint

Baseline Models
python model1.py
python model2.py

Experiments:
Experiment 1: Context Switching
python main.py --switch_period none --supervise all
python main.py --switch_period 32 --supervise all
python main.py --switch_period 64 --supervise all
python main.py --switch_period 128 --supervise all

Experiment 2: Supervision Granularity
python main.py --switch_period 64 --supervise all
python main.py --switch_period 64 --supervise last
python main.py --switch_period 64 --supervise query

Experiment 3: Model Capacity
Vary model size (width/depth) to test scaling effects.
# Example presets:
# Small  : d=64,  L=3, H=4
# Normal : d=128, L=4, H=4
# Medium : d=256, L=6, H=8
# Large  : d=512, L=8, H=8
python main.py --switch_period 64 --supervise all --d_model 64  --num_layers 3 --num_heads 4
python main.py --switch_period 64 --supervise all --d_model 128 --num_layers 4 --num_heads 4
python main.py --switch_period 64 --supervise all --d_model 256 --num_layers 6 --num_heads 8
python main.py --switch_period 64 --supervise all --d_model 512 --num_layers 8 --num_heads 8

Experiment 4: Training Stream Size
python main.py --switch_period 64 --supervise all --train_batches 250
python main.py --switch_period 64 --supervise all --train_batches 500
python main.py --switch_period 64 --supervise all --train_batches 1000
python main.py --switch_period 64 --supervise all --train_batches 2000


