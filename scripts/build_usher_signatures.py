#!/usr/bin/env python3
"""
Build a (mutation, lineage, weight) signatures table from the Usher barcode matrix.

Reads a lineage-by-mutation 0/1 matrix (for example Freyja's usher_barcodes.csv),
selects the configured lineages, renames each nucleotide mutation to its
amino-acid name where the change is nonsynonymous, and writes one row per
(mutation, lineage) pair with weight 1. A mutation shared by several lineages
appears once per lineage.

If the barcodes file is missing it is downloaded. An existing output file is
never overwritten unless --overwrite is given.

Settings come from the `signatures_usher` block in configs/default.yaml. Command
line flags override the config.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from utils.sars_cov2_aa import name_substitution, read_reference  # noqa: E402

DEFAULTS = {
    "barcodes_path": "data/usher_barcodes.csv",
    "reference_path": "../external/gromstole/data/NC_045512.fa",
    "url": "https://raw.githubusercontent.com/andersen-lab/Freyja/main/freyja/data/usher_barcodes.csv",
    "output_path": "data/signatures_usher_aa.csv",
    "lineages": ["B.1.1.7", "B.1.351", "B.1.617.2", "P.1"],
}

log = logging.getLogger("build_usher_signatures")


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


def load_config_block(config_path: Path) -> dict:
    import yaml

    if not config_path.exists():
        return {}
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    return dict(cfg.get("signatures_usher") or {})


def download(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Download to a temporary file in the destination directory, then move it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    log.info("Saved %s", dest)


def read_selected_mutations(barcodes: Path, lineages: Iterable[str]) -> "OrderedDict[str, list[str]]":
    """
    Return {lineage: [nucleotide mutations carried]} for the requested lineages.

    The first column of the matrix holds lineage names, the header holds mutation
    names. Values in the selected rows must be 0 or 1.
    """
    wanted = list(lineages)
    found: dict[str, list[str]] = {}
    with open(barcodes, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        muts = [h.strip() for h in header[1:]]
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if name not in wanted:
                continue
            if name in found:
                raise ValueError(f"lineage {name!r} appears more than once in {barcodes}")
            vals = row[1:]
            if len(vals) != len(muts):
                raise ValueError(f"row {name!r} has {len(vals)} values; header has {len(muts)} mutations")
            carried = []
            for m, v in zip(muts, vals):
                v = v.strip()
                if v == "1":
                    carried.append(m)
                elif v != "0":
                    raise ValueError(f"non-binary value {v!r} for lineage {name!r}, mutation {m!r}")
            found[name] = carried

    missing = [l for l in wanted if l not in found]
    if missing:
        raise ValueError(f"lineages not found in {barcodes}: {missing}")
    return OrderedDict((l, found[l]) for l in wanted)


def build_rows(selected: "OrderedDict[str, list[str]]", refseq: str) -> list[tuple[str, str]]:
    """Return sorted (mutation, lineage) pairs with nucleotide mutations renamed to amino-acid names."""
    rows: list[tuple[str, str]] = []
    n_untranslatable = 0
    for lineage, muts in selected.items():
        names: set[str] = set()
        for m in muts:
            try:
                names.add(name_substitution(m, refseq))
            except KeyError:
                n_untranslatable += 1
                names.add(m)
        n_dup = len(muts) - len(names)
        if n_dup:
            log.info("%s: %d nucleotide mutations collapsed to an existing amino-acid name", lineage, n_dup)
        log.info("%s: %d mutations", lineage, len(names))
        rows.extend((n, lineage) for n in sorted(names))
    if n_untranslatable:
        log.warning("%d mutations had codons with non-ACGT bases and kept their nucleotide names", n_untranslatable)
    return rows


def write_signatures(rows: list[tuple[str, str]], out: Path, overwrite: bool) -> None:
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(["mutation", "lineage", "weight"])
        for mutation, lineage in rows:
            w.writerow([mutation, lineage, 1])
    log.info("Wrote %d rows to %s", len(rows), out)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--barcodes", help="Usher barcode matrix (downloaded if missing)")
    ap.add_argument("--reference", help="NC_045512 FASTA")
    ap.add_argument("--url", help="URL used when the barcodes file is missing")
    ap.add_argument("--output", help="output signatures CSV")
    ap.add_argument("--lineages", nargs="+", help="lineage names to include")
    ap.add_argument("--overwrite", action="store_true", help="replace the output file if it exists")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = {**DEFAULTS, **load_config_block(_resolve(args.config))}
    barcodes = _resolve(args.barcodes or cfg["barcodes_path"])
    reference = _resolve(args.reference or cfg["reference_path"])
    output = _resolve(args.output or cfg["output_path"])
    lineages = args.lineages or list(cfg["lineages"])
    url = args.url or cfg["url"]

    if not barcodes.exists():
        download(url, barcodes)

    selected = read_selected_mutations(barcodes, lineages)
    refseq = read_reference(str(reference))
    rows = build_rows(selected, refseq)
    write_signatures(rows, output, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
