"""
Nucleotide-to-amino-acid naming for SARS-CoV-2 substitutions.

The codon table, ORF coordinates, and substitution logic below are adapted from
`SC2Locator` in `scripts/seq_utils.py` of gromstole
(https://github.com/PoonLab/gromstole, path external/gromstole). Changes from the
original: the code is a standalone function instead of a class, it takes the
reference sequence as an argument instead of reading it in the constructor, and
it converts the 1-indexed positions used in Usher names to 0-indexed positions.

MIT License

Copyright (c) 2022 PoonLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import re

# Codon table (gromstole, scripts/seq_utils.py: SC2Locator.gcode)
GCODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    '---': '-', 'XXX': '?'
}

# ORF coordinates on NC_045512, 0-indexed half-open (gromstole, scripts/seq_utils.py: SC2Locator.orfs).
# Insertion order matters: orf1a is checked before the overlapping orf1b.
ORFS = {
    'orf1a': (265, 13468),
    'orf1b': (13467, 21555),
    'S': (21562, 25384),
    'orf3a': (25392, 26220),
    'E': (26244, 26472),
    'M': (26522, 27191),
    'orf6': (27201, 27387),
    'orf7a': (27393, 27759),
    'orf7b': (27755, 27887),
    'orf8': (27893, 28259),
    'N': (28273, 29533),
    'orf10': (29557, 29674)
}

_SUB_RE = re.compile(r'^([ACGT])(\d+)([ACGT])$')


def name_substitution(mutation: str, refseq: str) -> str:
    """
    Return the amino-acid name for a nucleotide substitution, or the input name if
    the change is synonymous or outside any ORF.

    `mutation` uses 1-indexed reference coordinates, e.g. 'A23403G'. Amino-acid
    names follow gromstole's format, e.g. 'aa:S:D614G'. Raises ValueError if the
    name is malformed or the reference base does not match, and KeyError if the
    reference codon contains a non-ACGT base.
    """
    m = _SUB_RE.match(mutation)
    if not m:
        raise ValueError(f"not a nucleotide substitution: {mutation!r}")
    ref, pos1, alt = m.group(1), int(m.group(2)), m.group(3)
    pos = pos1 - 1
    if not 0 <= pos < len(refseq):
        raise ValueError(f"position out of range for reference: {mutation!r}")
    if refseq[pos] != ref:
        raise ValueError(
            f"reference base mismatch at {pos1}: expected {refseq[pos]!r}, got {ref!r}"
        )

    for orf, (left, right) in ORFS.items():
        if left <= pos < right:
            break
    else:
        return mutation

    codon_left = 3 * ((pos - left) // 3)
    codon_pos = (pos - left) % 3
    rcodon = refseq[left:right][codon_left:codon_left + 3]
    ramino = GCODE[rcodon]

    qcodon = list(rcodon)
    qcodon[codon_pos] = alt
    qamino = GCODE[''.join(qcodon)]

    if ramino == qamino:
        return mutation
    return f"aa:{orf}:{ramino}{1 + codon_left // 3}{qamino}"


def read_reference(path: str) -> str:
    """Read the first sequence of a FASTA file as an uppercase string."""
    seq_parts: list[str] = []
    seen_header = False
    with open(path) as handle:
        for line in handle:
            if line.startswith('>'):
                if seen_header:
                    break
                seen_header = True
                continue
            seq_parts.append(line.strip().upper())
    return ''.join(seq_parts)
