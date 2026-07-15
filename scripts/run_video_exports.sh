#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/phong/miniconda3/envs/2dgslam/bin/python}"
FPS="${RECORD_VIDEO_FPS:-1}"
LOG_DIR="$ROOT/log"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${VIDEO_EXPORT_LOG:-$LOG_DIR/video_export_${RUN_ID}.log}"

mkdir -p "$LOG_DIR"

echo "Video export run id: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "Python: $PYTHON_BIN"
echo "FPS: $FPS"

run_one() {
    local label="$1"
    local config="$2"

    echo "===== START $label $(date -Is) =====" | tee -a "$LOG_FILE"
    "$PYTHON_BIN" -u "$ROOT/slam.py" \
        --config "$config" \
        --record-optimization-video \
        --record-video-fps "$FPS" 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}
    echo "===== END $label status=$status $(date -Is) =====" | tee -a "$LOG_FILE"

    return "$status"
}

cd "$ROOT" || exit 1
run_one "tum_fr3_office_mono" "configs/mono/tum/fr3_office.yaml" || exit $?
run_one "replica_room1_mono" "configs/mono/replica/room1.yaml" || exit $?
