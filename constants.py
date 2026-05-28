"""Scalar constants for datasets and HDF5 visualization (no bundled asset paths)."""
import pathlib
import os

_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = os.environ.get('MIMOCO_DATA_DIR', str(_ROOT / 'data'))

DT = 0.02
FPS = 50
