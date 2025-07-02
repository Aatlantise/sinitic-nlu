#!/bin/bash
module load anaconda3/3.11
module load cuda/12.5
nvcc --version
source ../.virtualenv/sinitic/bin/activate
python3 -V

# needs to be run with source, not bash
