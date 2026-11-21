"""Resume a published Figshare/Zenodo file and verify its deposited MD5.

The partial file may contain holes. Only the saved prefix and completed byte
ranges may be read before the final checksum. Committed range metadata is
written atomically. A crash can lose the latest downloaded ranges but never
promotes unchecked bytes into a completed dataset.
"""

import argparse
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2] / "dataset_for_training"
CHUNK = 64 * 1024**2


def source(name):
    if name == "aimnet":
        d = ROOT / "AIMNet2_NSE/closed_shell_parent"
        meta = json.loads((d / "figshare-record.json").read_text())["files"]
        row = next(v for v in meta if v["name"] == "aimnet2_wb97m.h5")
        return d / row["name"], row["download_url"], row["size"], row["computed_md5"]
    if name == "ions16":
        d = ROOT / "AIMNet_NSE_2021"
        meta = json.loads((d / "zenodo-record.json").read_text())["files"]
        row = next(v for v in meta if v["key"] == "ions16.h5")
        return (
            d / row["key"],
            row["links"]["self"],
            row["size"],
            row["checksum"].removeprefix("md5:"),
        )
    d = ROOT / "RDB19_Rad"
    meta = json.loads((d / "zenodo-record.json").read_text())["files"]
    row = next(v for v in meta if v["key"] == "Reactions.tar.bz2")
    return (
        d / row["key"],
        row["links"]["self"],
        row["size"],
        row["checksum"].removeprefix("md5:"),
    )


def fetch(name, workers):
    dest, url, total, digest = source(name)
    chunk = 8 * 1024**2 if name == "ions16" else CHUNK
    part = dest.with_name(dest.name + ".part")
    journal = dest.parent / "download-progress.json"
    with (dest.parent / "download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if dest.exists():
            with dest.open("rb") as f:
                if hashlib.file_digest(f, "md5").hexdigest() != digest:
                    raise ValueError("Existing archive checksum differs from deposit")
            return
        state = (
            json.loads(journal.read_text())
            if journal.exists()
            else {"prefix": part.stat().st_size if part.exists() else 0, "complete": []}
        )
        if state.get("md5", digest) != digest or not 0 <= state["prefix"] <= total:
            raise ValueError("The transfer journal belongs to another deposit")
        state["md5"] = digest

        def save():
            tmp = journal.with_suffix(".pending")
            tmp.write_text(json.dumps(state))
            tmp.replace(journal)

        save()
        with closing(
            os.fdopen(os.open(part, os.O_RDWR | os.O_CREAT, 0o644), "r+b", buffering=0)
        ) as output:
            fd = output.fileno()

            def receive(lo):
                end = min(total, lo + chunk) - 1
                # A slow server can close a valid range early. Resume exactly
                # after the last complete local write, preserving the bytes
                # already received. Publish the range only after its last byte.
                pos = lo
                for retry in range(10):
                    try:
                        req = Request(
                            url + "?korny_range=" + str(pos),
                            headers={"Range": f"bytes={pos}-{end}"},
                        )
                        with urlopen(req, timeout=65) as f:
                            if (
                                f.status != 206
                                or f.headers["Content-Range"]
                                != f"bytes {pos}-{end}/{total}"
                            ):
                                raise ValueError("Incorrect HTTP range")
                            while pos <= end:
                                buf = f.read(min(262144, end + 1 - pos))
                                if not buf:
                                    raise OSError("Incomplete response")
                                if os.pwrite(fd, buf, pos) != len(buf):
                                    raise OSError("Short local write")
                                pos += len(buf)
                        return lo
                    except (OSError, ValueError, HTTPException) as exc:
                        print("Retry", pos, retry, str(exc), flush=True)
                        time.sleep(min(60, 2**retry))
                raise OSError(f"Could not retrieve range {lo} from {url}")

            complete = set(state["complete"])
            pending = [
                lo for lo in range(state["prefix"], total, chunk) if lo not in complete
            ]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for task in as_completed([pool.submit(receive, i) for i in pending]):
                    state["complete"].append(task.result())
                    save()
                    if len(state["complete"]) % 10 == 1:
                        done = state["prefix"] + sum(
                            min(chunk, total - i) for i in state["complete"]
                        )
                        print(
                            f"{dest.name} {done / 1e9:.2f}/{total / 1e9:.2f} GB",
                            flush=True,
                        )
            os.fsync(fd)
        with part.open("rb") as f:
            got = hashlib.file_digest(f, "md5").hexdigest()
        if got != digest or part.stat().st_size != total:
            raise ValueError(f"Final checksum mismatch for {dest}")
        part.replace(dest)
        journal.unlink()
        print(f"Verified {dest.name}, {total} bytes, MD5 {digest}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=["aimnet", "rdb19", "ions16"])
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    if not 1 <= args.workers <= 16:
        p.error("Choose between 1 and 16 transfer workers")
    fetch(args.source, args.workers)
