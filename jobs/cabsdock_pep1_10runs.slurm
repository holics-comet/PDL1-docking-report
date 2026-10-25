#!/bin/bash
#SBATCH -J CABS_Pep1
#SBATCH --comment="field=bio;appl=misc"
#SBATCH -p cpu
#SBATCH --array=1-10
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH -o /home01/%u/CABSdock_PD_L1/logs/cabsdock_%A_%a.out
#SBATCH -e /home01/%u/CABSdock_PD_L1/logs/cabsdock_%A_%a.err

set -euo pipefail

export PATH=/scratch/$USER/micromamba/envs/cabs/bin:$PATH

INPUT="/home01/$USER/CABSdock_PD_L1/inputs/PD_L1_cleaned.pdb"
OUTROOT="/scratch/$USER/CABSdock_PD_L1/runs"
PEPTIDE="CLQKTPKQC"

RUN=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
OUTDIR="${OUTROOT}/Pep1_run_${RUN}"

mkdir -p "$OUTDIR"

echo "=========================================="
echo "CABS-dock: PD-L1 + PD-L1 Pep-1"
echo "Job ID     : $SLURM_JOB_ID"
echo "Array ID   : $SLURM_ARRAY_TASK_ID"
echo "Node       : $(hostname)"
echo "Input      : $INPUT"
echo "Peptide    : $PEPTIDE"
echo "Output     : $OUTDIR"
echo "CABSdock   : $(which CABSdock)"
echo "=========================================="

CABSdock \
    -i "$INPUT" \
    -p "$PEPTIDE" \
    -w "$OUTDIR"

echo "Run ${RUN} finished successfully"
