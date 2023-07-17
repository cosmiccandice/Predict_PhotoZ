#!/bin/bash

#SBATCH --account=b1094
#SBATCH --partition=ciera-std
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --array=0-31%6
#SBATCH --time=01-00:00:00
#SBATCH --mem=5G
#SBATCH --job-name="ps1dr2_catalog_convert"
#SBATCH --output=/home/mcs8686/jobs/sbatch_ps1dr2_new_conversion_job.%A_%a.txt

. $HOME/.bashrc

# Name of a file containing arguments to pass to script in the end
# In this case, the file contains a list of file names (one file per line)
FILELIST=/projects/b1094/stroh/projects/panstarrs_photoz/file_list.txt

# Read the file and convert it to an array named $LINES
read -d '' -r -a LINES < $FILELIST

# Move to working directory if needed
cd /scratch/mcs8686

# Run script and pass one line to it each time
/projects/b1094/stroh/projects/panstarrs_photoz/ps1dr2_new_to_db.sh "${LINES[$SLURM_ARRAY_TASK_ID]}"
