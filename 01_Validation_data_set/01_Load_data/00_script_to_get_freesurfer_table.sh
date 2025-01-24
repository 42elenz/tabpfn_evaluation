#!/bin/bash

# Load FreeSurfer module
module load freesurfer/7.4.1

# Set directories
export SUBJECTS_DIR="/zi/flstorage/HITKIP/datasets/OpenNeuro/ds005237/derivatives/freesurfer/"
STATS_DIR="./results"     # output directory in current location

# Create results directory if it doesn't exist
mkdir -p "$STATS_DIR"

if [ -d "$SUBJECTS_DIR" ]; then
    # Get cortical parcellation stats (aparcstats2table)
    for MEAS in area volume thickness thicknessstd meancurv gauscurv foldind curvind
    do
        for HEMI in rh lh
        do
            aparcstats2table \
                --subjects `ls -d ${SUBJECTS_DIR}/sub-* | xargs -n 1 basename` \
                --hemi "$HEMI" \
                --meas "$MEAS" \
                --delimiter comma \
                --tablefile "${STATS_DIR}/${HEMI}.aparc.${MEAS}.csv"
        done
    done
    
    # Get subcortical segmentation stats (asegstats2table)
    for MEAS in volume mean
    do
        asegstats2table \
            --subjects `ls -d ${SUBJECTS_DIR}/sub-* | xargs -n 1 basename` \
            --meas "$MEAS" \
            --delimiter comma \
            --skip \
            --tablefile "${STATS_DIR}/aseg.${MEAS}.csv"
    done

    # Extract Euler numbers
    python stats2table/eno2table.py \
        --subjects `ls -d ${SUBJECTS_DIR}/sub-* | xargs -n 1 basename` \
        --tablefile "${STATS_DIR}/euler.csv" \
        --delimiter comma
fi