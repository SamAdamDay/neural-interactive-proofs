import os
from pathlib import Path

_file_dir = os.path.realpath(__file__)
_root_dir = Path(_file_dir).parent.parent.absolute()

DATA_DIR = os.path.join(_root_dir, "data")
GI_DATA_DIR = os.path.join(DATA_DIR, "graph_isomorphism")

RESULTS_DIR = os.path.join(_root_dir, "results")
GI_SOLO_AGENTS_RESULTS_DIR = os.path.join(RESULTS_DIR, "solo_gi_agents")