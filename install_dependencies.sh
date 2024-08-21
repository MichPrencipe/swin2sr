#!/bin/bash

# Update mamba and conda
mamba update mamba conda -y

# Install libraries
mamba install timm -c conda-forge -y
mamba install numpy -c conda-forge -y
mamba install pandas -c conda-forge -y
mamba install scipy -c conda-forge -y
mamba install matplotlib -c conda-forge -y

