#!/bin/bash
#BSUB -J cellranger_job            # LSF job name

for arg in "$@"; do
    echo "Argument: $arg"
done

