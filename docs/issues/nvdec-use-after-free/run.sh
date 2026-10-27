#!/usr/bin/env bash
# run.sh FILE [RUNS]: nvdec_uaf under fence.so in a fresh process each time,
# 6 cycles a process, until one faults or RUNS pass. The core goes where the
# system's core_pattern says; see README.md.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
FILE=${1:?an .mp4 with HEVC pictures}
RUNS=${2:-100}
ulimit -c unlimited
for r in $(seq "$RUNS"); do
    LD_PRELOAD=$HERE/fence.so "$HERE/nvdec_uaf" "$FILE" 6 --b-secs=10 > "run-$r.log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then
        echo "$(date +%T) run $r: exit $status — see run-$r.log and the core"
        exit 1
    fi
    echo "$(date +%T) run $r: 6 cycles, no fault"
done
echo "no fault in $RUNS runs"
