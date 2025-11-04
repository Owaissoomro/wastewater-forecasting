# utils/provenance.py — tiny stub to avoid import errors
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Mapping, Union
from pathlib import Path

PathLike = Union[str, Path]

def capture_environment() -> Dict[str, Any]:
    return {"provenance": "stub"}

def record_inputs_outputs(run_id: str, stage: str,
                          inputs: Optional[Union[PathLike, Sequence[PathLike], Mapping[str, PathLike]]] = None,
                          outputs: Optional[Union[PathLike, Sequence[PathLike], Mapping[str, PathLike]]] = None
                          ) -> Dict[str, Any]:
    return {"run_id": run_id, "stage": stage, "inputs": [], "outputs": []}