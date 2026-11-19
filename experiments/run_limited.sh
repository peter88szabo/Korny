#!/bin/sh
# Share one memory budget across Korny experiments and all their descendants.
# Use decimal bytes: 26 GB leaves 2 GB of headroom under the requested 28 GB.
set -eu
systemctl --user set-property --runtime kornyfits.slice \
    MemoryMax=26000000000 MemoryHigh=22000000000 MemorySwapMax=0
exec systemd-run --user --scope --quiet --slice=kornyfits.slice \
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"
