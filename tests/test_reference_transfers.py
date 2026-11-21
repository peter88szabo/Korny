"""Check transfer archive integrity, coordinate parsing and fragment bookkeeping."""

import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.full_public import download_references


class ReoptimizedRecordsTests(TestCase):
    def test_prediction_fields_do_not_displace_geometry_or_baseline(self):
        from experiments.full_public.amk_correction_data import xyz_frames

        frame = (
            b"2\nProperties=species:S:1:MACE_forces:R:3:pos:R:3 "
            b"energy_hf=-23 MACE_energy=7\n"
            b"C 90 91 92 0 0 0\nH 80 81 82 1.1 0 0\n"
        )
        ((metadata, z, r),) = xyz_frames(io.BytesIO(frame))
        np.testing.assert_array_equal(z, [6, 1])
        np.testing.assert_allclose(r, [[0, 0, 0], [1.1, 0, 0]])
        self.assertEqual(metadata["energy_hf"], "-23")

    def test_missing_product_fragment_retains_only_forward_barrier(self):
        from experiments.full_public.trial_reoptimized import complete_sides

        counts = np.array([[8, 4, 0, 2], [8, 4, 0, 2], [2, 1, 0, 0], [6, 3, 0, 2]])
        self.assertEqual(complete_sides([0], [2], counts, 1), [True, False])
        self.assertEqual(complete_sides([0], [2, 3], counts, 1), [True, True])


class RangeTests(TestCase):
    def test_short_response_resumes_next_byte_and_verifies_whole_file(self):
        data = b"0123456789abcdefghijklmnopqrstuvwxyz"
        calls = []

        def response(req, timeout):
            lo, hi = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
            calls.append((lo, hi))
            # The publisher announced the requested length, but its first
            # connection ended early. The next request must resume at byte 15.
            out = io.BytesIO(data[lo : 15 if len(calls) == 1 else hi + 1])
            out.status = 206
            out.headers = {"Content-Range": f"bytes {lo}-{hi}/{len(data)}"}
            return out

        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "sample.h5"
            url = "https://example.org/sample.h5"
            meta = dest, url, len(data), hashlib.md5(data).hexdigest()
            dest.with_suffix(".h5.part").write_bytes(data[:7])
            with (
                mock.patch.object(download_references, "source", return_value=meta),
                mock.patch.object(download_references, "urlopen", side_effect=response),
                mock.patch.object(download_references.time, "sleep"),
            ):
                download_references.fetch("aimnet", 1)
            self.assertEqual(calls, [(7, len(data) - 1), (15, len(data) - 1)])
            self.assertEqual(dest.read_bytes(), data)
            self.assertFalse((dest.parent / "download-progress.json").exists())
