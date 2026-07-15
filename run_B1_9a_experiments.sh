#!/bin/bash
# Run B1-9a drift experiments 1 and 2 sequentially
export PYTHONNOUSERSITE=1
cd /home/2DGS_SLAM/2dgslam || exit 1
mkdir -p log

echo "=========================================================="
echo "B1-9a DRIFT EXPERIMENTS"
echo "=========================================================="

# Experiment 1: tracking_itr_num=300
echo ""
echo "=========================================================="
echo ">>> EXP 1: Tracking Iterations = 300 | $(date)"
echo "=========================================================="
/home/phong/miniconda3/envs/2dgslam/bin/python -u slam.py \
    --config configs/mono/tum/B1-9a_exp1.yaml \
    --eval \
    --record-optimization-video
status=$?
echo ">>> EXP 1 finished with status $status at $(date)"
sleep 5

# Experiment 2: aggressive keyframing
echo ""
echo "=========================================================="
echo ">>> EXP 2: Aggressive Keyframing (kf_interval=3, kf_translation=0.04) | $(date)"
echo "=========================================================="
/home/phong/miniconda3/envs/2dgslam/bin/python -u slam.py \
    --config configs/mono/tum/B1-9a_exp2.yaml \
    --eval \
    --record-optimization-video
status=$?
echo ">>> EXP 2 finished with status $status at $(date)"

echo ""
echo "=========================================================="
echo "ALL EXPERIMENTS COMPLETED $(date)"
echo "=========================================================="
