"""Compute the two small, held-out harmonic references with bare PBE0.

All Behemoth subprocesses, input, output, scratch and optimizer files live on
the designated run disk. Results are never fed to the force-field optimizer.
RKS uses exact exchange without empirical dispersion. Each system is optimized
before its analytical gradients are differentiated for the harmonic Hessian.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

ROOT = Path("/run/media/peter/Run/Behemoth")
CODE = Path("/home/peter/DevCode/Behemoth")
BINARY = CODE/"target/release/behemoth"
XYZ = CODE/"validation/ynk_kernel_benchmark/pbe0_reference/tz"
RUN = ROOT/"Korny_frequency_validation"/"PBE0_cc-pVTZ_v1"
INPUT = """nproc = 2
memory = 2048
method = RKS PBE0
basis = cc-pVTZ
todo = Opt Freq
charge = 0
multi = 1
xyzfile = geometry.xyz

@grid
  level = fine
  pruning = tight_adaptive
end
@scf
  mixer = diis
  max_iterations = 200
  energy_threshold = 1e-11
  density_threshold = 1e-10
  gradient_threshold = 1e-9
end
"""


def main():
    """Keep a unique attempt directory and skip previously completed jobs with identical inputs."""
    if not ROOT.is_dir() or not os.path.ismount("/run/media/peter/Run"):
        raise RuntimeError("Requested calculation disk is not mounted")
    binary_hash = hashlib.sha256(BINARY.read_bytes()).hexdigest()
    for label, xyz in [("water", "g2rc_13.xyz"), ("ammonia", "g2rc_11.xyz")]:
        geom = (XYZ/xyz).read_text()
        key = hashlib.sha256((binary_hash+INPUT+geom).encode()).hexdigest()
        root = RUN/label
        root.mkdir(parents=True, exist_ok=True)
        if any(json.loads(s.read_text()).get("completed_key")==key for s in root.glob("attempt-*/status.json")):
            continue
        attempt = root/("attempt-"+str(len(list(root.glob("attempt-*")))+1))
        attempt.mkdir()
        (attempt/"geometry.xyz").write_text(geom)
        (attempt/"frequency.inp").write_text(INPUT)
        scratch = attempt/"scratch"
        scratch.mkdir()
        env = dict(os.environ, TMPDIR=str(scratch), OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="2")
        start = monotonic()
        with (attempt/"behemoth.out").open("w") as log:
            run = subprocess.run([str(BINARY), "frequency.inp"], cwd=attempt, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        output = (attempt/"behemoth.out").read_text()
        done = run.returncode == 0 and "freq[cm-1]" in output
        (attempt/"status.json").write_text(json.dumps(dict(returncode=run.returncode,
                 seconds=monotonic()-start, completed_key=key if done else None, binary_sha256=binary_hash),indent=2)+"\n")
        print(label,done,attempt,flush=True)
        if not done:
            raise RuntimeError(f"Check {attempt/'behemoth.out'}")


if __name__ == "__main__":
    main()
