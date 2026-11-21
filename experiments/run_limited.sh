#!/bin/sh
# Share one memory budget across Korny experiments and all their descendants.
# Account for mapped training files and all workers, as well as private arrays.
# Reclaim at 6 GB and leave room for short allocation bursts below 8 GB.
set -eu
systemctl --user set-property --runtime kornyfits.slice \
    MemoryMax=8000000000 MemoryHigh=6000000000 MemorySwapMax=0
exec systemd-run --user --scope --quiet --slice=kornyfits.slice \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"
