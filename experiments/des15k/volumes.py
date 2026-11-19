"""Fit the independent TS volume model using only the DES training partition."""
import hashlib
from io import BytesIO
import json
from time import perf_counter

import h5py

from .run import DATA, OUT, SPECIES
from src.dispersion.volumes import HirshfeldVolumes


def fit_volumes():
    heldout = set((OUT/"test_parents.txt").read_text().splitlines())
    training_count = 0
    def records(h):
        nonlocal training_count
        for parent in h.values():
            for name, g in parent.items():
                if name.rsplit("-", 1)[0] in heldout:
                    continue
                training_count += 1
                yield g["atNUM"][:], g["atXYZ"][:], g["hRAT"][:]
    spec = dict(species=SPECIES, nr=2, lmax=2, ron=2., rc=3.8)
    started = perf_counter()
    with h5py.File(DATA/"DES15k.hdf5") as h:
        model = HirshfeldVolumes.fit(records(h), spec)
    elapsed = perf_counter()-started
    expected = json.loads((OUT/"results.json").read_text())["training"]["structures"]
    if expected != training_count:
        raise ValueError("Volume and energy training sets differ")
    buffer = BytesIO()
    model.save(buffer)
    blob = buffer.getvalue()
    name = "volumes-" + hashlib.sha256(blob).hexdigest()[:16] + ".npz"
    (OUT/name).write_bytes(blob)
    report = dict(checkpoint=name, target="log(hRAT)", training=training_count,
                  heldout_parents=len(heldout), spec=spec, fit_seconds=elapsed)
    (OUT/"volumes.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    fit_volumes()
