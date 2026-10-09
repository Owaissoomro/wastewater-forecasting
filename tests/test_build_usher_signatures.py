"""Tests for the Usher signature builder (scripts/build_usher_signatures.py, utils/sars_cov2_aa.py)."""

import csv
import importlib.util
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from utils.sars_cov2_aa import name_substitution  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_usher_signatures", _REPO / "scripts" / "build_usher_signatures.py"
)
bus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bus)

S_GENE_START = 21562  # 0-indexed, gromstole orfs['S']
REF_LEN = 29903


@pytest.fixture
def refseq():
    seq = ["A"] * REF_LEN
    # codon 1 of S: GCT (Ala); third base T at 1-indexed 21565
    seq[S_GENE_START:S_GENE_START + 3] = list("GCT")
    # codon 10 of S: TGG (Trp); middle G at 1-indexed 21591
    seq[S_GENE_START + 27:S_GENE_START + 30] = list("TGG")
    # 5' UTR base, outside any ORF, 1-indexed 101
    seq[100] = "C"
    return "".join(seq)


def test_nonsynonymous_change_gets_aa_name(refseq):
    assert name_substitution("G21591C", refseq) == "aa:S:W10S"


def test_synonymous_change_keeps_nucleotide_name(refseq):
    assert name_substitution("T21565C", refseq) == "T21565C"


def test_non_orf_change_keeps_nucleotide_name(refseq):
    assert name_substitution("C101T", refseq) == "C101T"


def test_reference_mismatch_raises(refseq):
    with pytest.raises(ValueError, match="reference base mismatch"):
        name_substitution("A101T", refseq)


def test_malformed_name_raises(refseq):
    with pytest.raises(ValueError):
        name_substitution("del:1:3", refseq)


def _write_matrix(path, rows, muts):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow([""] + muts)
        for name, vals in rows:
            w.writerow([name] + vals)


@pytest.fixture
def matrix(tmp_path):
    p = tmp_path / "barcodes.csv"
    muts = ["G21591C", "T21565C", "C101T", "A500G"]
    _write_matrix(p, [
        ("L1", ["1", "1", "0", "0"]),
        ("L2", ["1", "0", "1", "0"]),
        ("L3", ["0", "0", "0", "1"]),
    ], muts)
    return p


def test_selected_lineages_keep_carriers_in_config_order(matrix):
    sel = bus.read_selected_mutations(matrix, ["L2", "L1"])
    assert list(sel) == ["L2", "L1"]
    assert sel["L2"] == ["G21591C", "C101T"]
    assert sel["L1"] == ["G21591C", "T21565C"]


def test_missing_lineage_raises(matrix):
    with pytest.raises(ValueError, match="not found"):
        bus.read_selected_mutations(matrix, ["L1", "NOPE"])


def test_non_binary_value_raises(tmp_path):
    p = tmp_path / "bad.csv"
    _write_matrix(p, [("L1", ["2", "0"])], ["A1C", "C2T"])
    with pytest.raises(ValueError, match="non-binary"):
        bus.read_selected_mutations(p, ["L1"])


def test_build_rows_renames_and_keeps_shared_mutations(refseq, matrix):
    sel = bus.read_selected_mutations(matrix, ["L1", "L2"])
    rows = bus.build_rows(sel, refseq)
    assert ("aa:S:W10S", "L1") in rows
    assert ("aa:S:W10S", "L2") in rows
    assert rows.count(("aa:S:W10S", "L1")) == 1
    assert ("T21565C", "L1") in rows
    assert ("C101T", "L2") in rows
    assert len(rows) == 4


def test_main_writes_output_and_refuses_overwrite(tmp_path, refseq, matrix, monkeypatch):
    ref = tmp_path / "ref.fa"
    ref.write_text(">NC_045512\n" + "\n".join(refseq[i:i + 60] for i in range(0, REF_LEN, 60)) + "\n")
    out = tmp_path / "out.csv"
    args = ["--config", str(tmp_path / "none.yaml"), "--barcodes", str(matrix),
            "--reference", str(ref), "--output", str(out), "--lineages", "L1"]

    assert bus.main(args) == 0
    with open(out) as fh:
        got = list(csv.DictReader(fh))
    assert [r["mutation"] for r in got] == ["T21565C", "aa:S:W10S"]
    assert {r["weight"] for r in got} == {"1"}

    with pytest.raises(FileExistsError):
        bus.main(args)


def test_download_only_when_barcodes_missing(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dest, timeout=120.0):
        calls.append((url, dest))
        _write_matrix(dest, [("L1", ["1"])], ["A1C"])

    monkeypatch.setattr(bus, "download", fake_download)
    ref = tmp_path / "ref.fa"
    ref.write_text(">x\n" + "A" * 100 + "\n")
    base = ["--config", str(tmp_path / "none.yaml"), "--reference", str(ref),
            "--lineages", "L1"]

    missing = tmp_path / "absent.csv"
    bus.main(base + ["--barcodes", str(missing), "--output", str(tmp_path / "o1.csv")])
    assert len(calls) == 1 and calls[0][1] == missing

    present = tmp_path / "present.csv"
    _write_matrix(present, [("L1", ["1"])], ["A1C"])
    bus.main(base + ["--barcodes", str(present), "--output", str(tmp_path / "o2.csv")])
    assert len(calls) == 1
