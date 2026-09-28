#!/bin/bash
set -e
export OMP_NUM_THREADS=1
mkdir -p out tmp
export TMPDIR=$(pwd)/tmp
# dry-run: graphene/hBN 30deg forced-strain, no SOC, k=4x4x1
mpirun -np 8 pw.x -in scf.in | tee scf.out
mpirun -np 8 pw.x -in nscf.in | tee nscf.out
wannier90.x -pp wannier90
mpirun -np 8 pw2wannier90.x -in pw2wan.in | tee pw2wan.out
wannier90.x wannier90
echo DRYRUN_COMPLETE
