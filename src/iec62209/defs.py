# ==============================================================================
# utilitaries

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).absolute().parent.parent

ROOT_PATH = project_root()
SRC_PATH = os.path.join(ROOT_PATH, 'src')
LIB_PATH = os.path.join(ROOT_PATH, 'lib')
OUT_PATH = os.path.join(ROOT_PATH, 'output')
DATA_PATH = os.path.join(ROOT_PATH, 'data')

sys.path.append(SRC_PATH)
sys.path.append(LIB_PATH)
