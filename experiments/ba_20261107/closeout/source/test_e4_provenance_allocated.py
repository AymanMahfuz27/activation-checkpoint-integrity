"""Allocated-only regression controls for E4's source hash representation."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from analysis_core import require_allocation
from e4_provenance import CUDA, HEADER, LOADER, INCLUDE, reconstruct_extension_source


def source_fixture(root):
    """Independent literal source and expected expansion, without executing code."""
    root = Path(root).resolve() / "frozen-source"
    root.mkdir()
    contents = {
        CUDA: b'// synthetic CUDA\n#include "blake3_core.h"\nvoid kernel() {}\n',
        HEADER: b"// synthetic header\nint independent_header = 7;\n",
        LOADER: b"# synthetic loader identity only; no code is imported\n",
    }
    for member, content in contents.items():
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    original = {
        "source_path": str(root),
        "source_files": {
            member: hashlib.sha256(content).hexdigest()
            for member, content in contents.items()
        },
    }
    expected = hashlib.sha256(
        b"// synthetic CUDA\n// synthetic header\nint independent_header = 7;\n\nvoid kernel() {}\n"
    ).hexdigest()
    return original, expected


class E4ProvenanceFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_exact_header_expansion_matches_independent_literal(self):
        with tempfile.TemporaryDirectory() as folder:
            original, expected = source_fixture(folder)
            actual = reconstruct_extension_source(original)
            self.assertEqual(actual["assembled_nvrtc_source_sha256"], expected)
            self.assertNotEqual(actual["raw_cuda_source_sha256"], expected)

    def test_wrong_header_raw_cuda_or_loader_bytes_are_rejected(self):
        for member in (HEADER, CUDA, LOADER):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as folder:
                original, _ = source_fixture(folder)
                path = Path(original["source_path"]) / member
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "source byte hash differs"):
                    reconstruct_extension_source(original)

    def test_missing_header_or_missing_pin_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            original, _ = source_fixture(folder)
            unpinned = copy.deepcopy(original)
            del unpinned["source_files"][HEADER]
            with self.assertRaisesRegex(ValueError, "source byte hash differs"):
                reconstruct_extension_source(unpinned)
            (Path(original["source_path"]) / HEADER).unlink()
            with self.assertRaisesRegex(ValueError, "regular file"):
                reconstruct_extension_source(original)

    def test_missing_or_duplicate_include_is_not_an_alternative_namespace(self):
        for count in (0, 2):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder:
                original, _ = source_fixture(folder)
                path = Path(original["source_path"]) / CUDA
                raw = INCLUDE * count
                path.write_bytes(raw)
                original["source_files"][CUDA] = hashlib.sha256(raw).hexdigest()
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    reconstruct_extension_source(original)

    def test_runtime_raw_hash_wrong_expanded_hash_and_architecture_are_rejected(self):
        from subset_analysis import e4_extra_validation
        from test_subset_integration_allocated import population

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readers, _, _, _, _ = population(root)
            from evidence_loading import load_e4

            raw = load_e4(readers)
            original = readers["3520049"].original_source_contract
            e4_extra_validation(raw, readers, original)
            mutations = (
                {"source_sha256": original["source_files"][CUDA]},
                {"source_sha256": "0" * 64},
                {"architecture": "sm_90"},
            )
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    wrong = copy.deepcopy(raw)
                    wrong["setup"]["3520050"]["extension"].update(mutation)
                    with self.assertRaisesRegex(
                        ValueError, "extension lineage differs"
                    ):
                        e4_extra_validation(wrong, readers, original)


if __name__ == "__main__":
    unittest.main()
