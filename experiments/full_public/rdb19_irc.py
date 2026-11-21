"""Import gas-phase radical IRC paths from the published Gaussian logs.

Read the bzip2 tar sequentially. Only accepted path points enter the model
archive. Coordinates, SCF energies and forces must belong to the same step,
and the ordered atom IDs in the two Gaussian tables must match. Preserve the
actual Gaussian method and path direction. The names "forw" and "rev" in
these files do not identify the educt side of the published reaction table.
"""

import argparse
import json
import re
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from rdkit import Chem

from experiments.full_public.data import DATA
from experiments.full_public.download_references import CHUNK
from src.units import BOHR_A, HARTREE_EV

ROOT = DATA / "RDB19_Rad"
OUT = ROOT / "electronic-irc"
STATE = re.compile(r"Charge\s*=\s*([+-]?\d+)\s+Multiplicity\s*=\s*(\d+)")
POINT = re.compile(r"Point Number:\s*(\d+)\s+Path Number:\s*(\d+)")
ENERGY = re.compile(r"SCF Done:\s+E\(([^)]+)\)\s*=\s*([-.\dDE+]+)")
MEMBER = re.compile(r"Reactions/(\d+)/irc_gaussian_(forw|rev)\.log$")


def read_log(lines):
    """Keep accepted IRC geometries, then verify the independently printed path energies."""
    z = state = xyz = energy = force = method = None
    points, coords, energies, forces, summary = [], [], [], [], []
    route = ""
    clean = False
    for line in lines:
        s = line.strip()
        if s.startswith("#") and not route and not s.startswith("# OF"):
            route = s
            for ln in lines:
                if ln.strip().startswith("-----"):
                    break
                route += ln.strip()
        elif "Charge =" in line and (match := STATE.search(line)):
            state = tuple(map(int, match.groups()))
        elif s in ("Input orientation:", "Standard orientation:"):
            # Gaussian forces use standard orientation when printed. The
            # source calculations use NoSymmetry and print the input frame.
            next(lines)
            next(lines)
            next(lines)
            buf = []
            for ln in lines:
                if ln.strip().startswith("-----"):
                    if not buf:
                        continue
                    break
                buf.append(ln.split())
            if not buf or any(int(a[0]) != i + 1 for i, a in enumerate(buf)):
                raise ValueError("Malformed coordinate table")
            here = np.array([a[1] for a in buf], np.uint8)
            if z is not None and not np.array_equal(z, here):
                raise ValueError("Atom order changed")
            z = here
            xyz = np.array([a[3:6] for a in buf], float)
            energy = force = None
        elif "SCF Done:" in line:
            match = ENERGY.search(line)
            if not match:
                raise ValueError("Unrecognized SCF energy")
            method, energy = match[1], float(match[2].replace("D", "E"))
        elif "Forces (Hartrees/Bohr)" in line and "Center" in line:
            next(lines)
            sep = next(lines).strip()
            if not sep.startswith("-----"):
                raise ValueError("Malformed force table")
            buf = []
            for ln in lines:
                if ln.strip().startswith("-----"):
                    break
                buf.append(ln.split())
            if (
                z is None
                or len(buf) != len(z)
                or any(
                    int(a[0]) != i + 1 or int(a[1]) != z[i] for i, a in enumerate(buf)
                )
            ):
                raise ValueError("Force atom indices differ from geometry")
            force = np.array([a[2:5] for a in buf], float)
        elif "Point Number:" in line and (m := POINT.search(line)):
            if state is None or xyz is None or force is None or energy is None:
                raise ValueError("Accepted IRC point lacks a matching force or energy")
            points.append(int(m[1]))
            coords.append(xyz)
            forces.append(force)
            energies.append(energy)
            xyz = energy = force = None
        elif "Summary of reaction path following" in line:
            # The reverse-path summary runs from the last point back to the TS.
            for ln in lines:
                v = ln.split()
                if len(v) == 3 and v[0].isdigit():
                    summary.append(float(v[1]))
                elif summary:
                    break
        elif "Normal termination of Gaussian" in line:
            clean = True
    if not clean or not points:
        return None
    if (
        not route
        or "nosym" not in route.lower()
        or len(points) != len(set(points))
        or points[0] != 0
    ):
        raise ValueError("Unexpected IRC convention")
    data = np.array(energies)
    if len(summary) == len(data):
        d = data - data[0]
        if min(np.max(np.abs(d - summary)), np.max(np.abs(d[::-1] - summary))) > 6.1e-6:
            raise ValueError("SCF labels disagree with Gaussian's IRC energy summary")
    else:
        raise ValueError("Missing accepted-point energy cross-check")
    if (int(z.sum()) - state[0] - (state[1] - 1)) % 2:
        raise ValueError("Spin and electron-count parity disagree")
    f = np.asarray(forces)
    if not np.isfinite(f).all() or not np.isfinite(data).all():
        raise ValueError("Nonfinite labels")
    return {
        "z": z,
        "r_A": np.array(coords, dtype=np.float32),
        "energy_eV": data * HARTREE_EV,
        "force_eV_A": (f * (HARTREE_EV / BOHR_A)).astype(np.float32),
        "point": points,
        "charge": state[0],
        "multiplicity": state[1],
        "scf": method,
        "route": route,
    }


class GrowingArchive:
    """Allow a sequential decoder to read only committed contiguous HTTP ranges."""

    def __init__(self, file, final):
        self.file = file
        self.final = final
        self.journal = final.parent / "download-progress.json"
        self.limit = 0

    def read(self, size):
        while self.file.tell() + size > self.limit:
            if self.final.exists():
                return self.file.read(size)
            try:
                state = json.loads(self.journal.read_text())
            except FileNotFoundError:
                if self.final.is_file():
                    return self.file.read(size)
                raise
            cursor = state["prefix"]
            complete = set(state["complete"])
            while cursor in complete:
                cursor += CHUNK
            self.limit = cursor
            if self.file.tell() + size > self.limit:
                time.sleep(3)
        return self.file.read(size)


@contextmanager
def stream(path, follow):
    if follow and not path.exists():
        with (
            path.with_name(path.name + ".part").open("rb") as f,
            tarfile.open(
                fileobj=GrowingArchive(f, path), mode="r|bz2", bufsize=1024**2
            ) as t,
        ):
            yield t
    else:
        with tarfile.open(path, "r|bz2", bufsize=1024**2) as t:
            yield t


def endpoint(t, member, folder):
    """Preserve the authors' actual educt/product endpoint complexes and TS atom labels."""
    m = re.fullmatch(r"Reactions/(\d+)/(IRC_educt|IRC_product|ts)\.xyz", member.name)
    if not m:
        return
    dest = folder / f"{m[1]}-{m[2]}.npz"
    if dest.exists():
        return
    with t.extractfile(member) as f:
        lines = f.read().decode("ascii", errors="replace").splitlines()
    n = int(lines[0])
    vals = [v.split() for v in lines[2 : 2 + n]]
    if len(vals) != n:
        raise ValueError(member.name)
    pt = Chem.GetPeriodicTable()
    z = np.array([pt.GetAtomicNumber(v[0].capitalize()) for v in vals], np.uint8)
    r = np.array([v[1:4] for v in vals], float)
    center = re.search(r'reaction_center="_JSON (\[[^\]]*\])"', lines[1])
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(
            f,
            z=z,
            r_A=r,
            comment=lines[1],
            center=json.loads(center[1]) if center else [],
        )
    dest.with_suffix(".pending").replace(dest)


def import_archive(path, limit=None, follow=False):
    """Write one atomic, portable archive per complete log. Reruns reuse existing paths."""
    OUT.mkdir(exist_ok=True)
    counts = {"imported": 0, "incomplete": 0, "unrecognized": 0, "geometries": 0}
    errors = {}
    endpoints = OUT / "source-endpoints"
    endpoints.mkdir(exist_ok=True)
    with stream(path, follow) as t:
        for member in t:
            if member.name.endswith(".xyz"):
                endpoint(t, member, endpoints)
            m = MEMBER.fullmatch(member.name)
            if not m:
                continue
            dest = OUT / f"{m[1]}-{m[2]}.npz"
            if dest.exists():
                continue
            with t.extractfile(member) as f:
                try:
                    result = read_log(
                        line.decode("ascii", errors="replace") for line in f
                    )
                except (ValueError, IndexError, StopIteration) as exc:
                    errors[member.name] = str(exc)
                    counts["unrecognized"] += 1
                    continue
            if result is None:
                counts["incomplete"] += 1
                continue
            with dest.with_suffix(".pending").open("wb") as f:
                np.savez_compressed(f, rxn_id=int(m[1]), direction=m[2], **result)
            dest.with_suffix(".pending").replace(dest)
            counts["imported"] += 1
            counts["geometries"] += len(result["point"])
            if counts["imported"] % 25 == 0:
                print("Verified Gaussian IRC logs", counts, flush=True)
            if limit and counts["imported"] >= limit:
                break
    print("IRC import", counts, flush=True)
    (OUT / "import-summary.json").write_text(
        json.dumps({"pass": counts, "exceptions": errors}, indent=2)
    )
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ROOT / "Reactions.tar.bz2")
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--follow",
        action="store_true",
        help="Read committed ranges as a running transfer makes them available",
    )
    args = p.parse_args()
    import_archive(args.archive, args.limit, args.follow)
