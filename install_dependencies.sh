#!/bin/bash

# Update mamba and conda
mamba update mamba conda -y

# Install libraries
mamba install timm -c conda-forge -y
mamba install numpy -c conda-forge -y
mamba install pandas -c conda-forge -y
mamba install scipy -c conda-forge -y
mamba install matplotlib -c conda-forge -y
# Install OpenCV and scikit-image
mamba install opencv -c conda-forge -y
mamba install scikit-image -c conda-forge -y
mamba install transformers -c conda-forge -y

