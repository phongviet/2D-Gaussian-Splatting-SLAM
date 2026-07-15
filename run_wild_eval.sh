#!/bin/bash
export PYTHONNOUSERSITE=1
cd /home/2DGS_SLAM/2dgslam || exit 1

mkdir -p log

SEQS=("Bedroom_desk" "Studyroom_desk" "Studyroom")

echo "=========================================================="
echo "STARTING WILD EVALUATION (MONOCULAR + VIDEO RECORDING)"
echo "=========================================================="

for seq in "${SEQS[@]}"; do
    echo "=========================================================="
    echo ">>> Running Sequence: TUM wild/$seq | $(date)"
    echo "=========================================================="
    
    /home/phong/miniconda3/envs/2dgslam/bin/python -u slam.py \
        --config "configs/mono/tum/${seq}.yaml" \
        --eval \
        --record-optimization-video
        
    status=$?
    if [ $status -eq 0 ]; then
        echo ">>> Completed successfully: TUM $seq"
    else
        echo ">>> FAILED: TUM $seq with status $status"
    fi
    echo ""
    sleep 3
done

echo "=========================================================="
echo "ALL EVALUATIONS COMPLETED $(date)"
echo "=========================================================="
