"""Fit H-transfer energy, projected forces and saddle curvature on RDB19-Rad.

The shared p67 native potential is frozen. Each H kernel gets one readout for
all 4,974 training reactions. All 626 withheld reactions retain their original
orientation and native electronic reference; no reaction class enters inference.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public.audit_irc_curvature import (
    path_chord,
    signed_wavenumber,
    TO_CM1,
)
from experiments.full_public.h_transfer_sources import RADICAL, ROOT, FAMILIES, IRC
from experiments.full_public.irc_learning_curve import direction
from experiments.full_public.native_irc_pes import NativeRadicalPotential
from experiments.full_public.trial_h_transfer import bases, dual_features
from experiments.full_public.trial_path_scales import observation_rows, metrics


def encode(rec):
    rx = rec["rxn"]

    def features(pts):
        both = dual_features([(z, r.astype(float)) for z, r in pts], models)
        return np.column_stack([both[f] for f in FAMILIES])

    original = observation_rows(rx, features)
    with np.load(IRC.parent / "multiscale-path61" / f"{rx}.npz") as f:
        old = f["rows"].astype(float)
    np.testing.assert_allclose(original[:, -1], old[:, -1], rtol=1e-5, atol=3e-5)
    y = original[:, -1] - old[:, :-1] @ parent_weights
    _, chord = path_chord(rx)
    curvature = np.zeros(2 * models[FAMILIES[0]].size)
    if chord is not None:
        z, coords, forces, v, length = chord
        # The derivative of each energy basis gives its force on the two
        # flanks. The difference along the mass-weighted chord gives curvature.
        for j, f in enumerate(FAMILIES):
            b = models[f]
            left, right = [v.ravel() @ b.design(z, r, True)[1] for r in coords]
            curvature[j * b.size : (j + 1) * b.size] = (left - right) / length
    sign = 1
    if rec["heldout"]:
        sign = 1 if direction(rx)[0] == 0 else -1
    return original[:, :-1], y, curvature, sign


def run():
    global models, parent_weights
    models = bases()
    records = json.loads((RADICAL / "snapshot.json").read_text())
    with np.load(RADICAL / "native.npz") as f:
        parent_weights = f["weights"][:, 0]
    train = np.array([not r["heldout"] for r in records])
    h = np.array(
        [r["family"] in ("H Atom Abstraction", "intra_H_migration") for r in records]
    )
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(encode, records, chunksize=4, buffersize=16))
    x, y, c, sign = map(np.array, zip(*rows))
    del rows
    old_curv = json.loads((RADICAL / "held-path-curvature.json").read_text())["rows"]
    assert [r["rxn"] for i, r in enumerate(records) if not train[i]] == [
        v["id"] for v in old_curv
    ]
    curves = np.array(
        [[v["signed_cm1"][k] for k in ("reference", "67")] for v in old_curv]
    )
    cv = np.sign(curves) * (curves / TO_CM1) ** 2
    with np.load(IRC.parent / "multiscale-path66/training-curvatures.npz") as f:
        assert f["ids"].tolist() == [
            r["rxn"] for i, r in enumerate(records) if train[i]
        ]
        use = f["use"]
        raw = f["rows"]
        yc = raw[:, -1] - raw[:, :-1] @ parent_weights
    native = NativeRadicalPotential.load(RADICAL / "native-radical-pes.npz")
    scores = {}
    old_ref = json.loads((RADICAL / "all-held-validation.json").read_text())[0][
        "shared"
    ]
    for family, a, b in (
        (None, None, None),
        *(
            (f, j * models[f].size, (j + 1) * models[f].size)
            for j, f in enumerate(FAMILIES)
        ),
    ):
        change = np.zeros_like(y)
        dh = np.zeros(len(cv))
        if family:
            bank = models[family]
            # Retain all path energy/force observations. A fixed ridge keeps
            # this correction local to the existing radical PES. The factor
            # 0.006 regularizes changes in measured saddle curvature.
            z = x[train, :, a:b].reshape(-1, bank.size)
            d = c[train, a:b][use]
            mat = z.T @ z + 0.006 * d.T @ d + 60 * np.eye(bank.size)
            w = cho_solve(
                cho_factor(mat), z.T @ y[train].ravel() + 0.006 * d.T @ yc[use]
            )
            bank.weights = w
            change = x[:, :, a:b] @ w
            dh = c[~train, a:b] @ w
            native.hydrogen_transfer = bank
            dest = ROOT / f"native-rdb19-{family}.npz"
            native.save(dest)
            restored = NativeRadicalPotential.load(dest)
            with np.load(IRC / f"{records[0]['rxn']}-forw.npz") as d:
                zz, rr = d["z"], d["r_A"][0].astype(float)
            v = np.random.default_rng(86).normal(size=rr.shape)
            v /= np.linalg.norm(v)
            fd = (
                restored.energy(zz, rr - 2e-5 * v) - restored.energy(zz, rr + 2e-5 * v)
            ) / 4e-5
            np.testing.assert_allclose(
                fd, (v * restored.predict(zz, rr)[1]).sum(), rtol=0, atol=4e-5
            )
        err = change - y
        panels = {}
        for label, mask in (("all", ~train), ("H transfers", ~train & h)):
            sub = mask[~train]
            freq = signed_wavenumber(cv[sub, 1] + dh[sub])
            panels[label] = metrics(err[mask, 0] * sign[mask], err[mask, 1]) | dict(
                n=int(mask.sum()),
                signed_cm1=float(abs(freq - curves[sub, 0]).mean()),
                downhill=int(np.sum(freq < 0)),
            )
        if family is None:
            for k in ("reaction", "forward", "reverse", "barriers"):
                np.testing.assert_allclose(panels["all"][k], old_ref[k], atol=2e-5)
        scores[family or "p67"] = panels
        print("Native doublet", family or "p67", panels, flush=True)
    kt.write_json(ROOT / "native-rdb19-validation.json", scores)


if __name__ == "__main__":
    run()
