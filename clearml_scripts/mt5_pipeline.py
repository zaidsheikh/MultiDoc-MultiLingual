from pathlib import Path
print(Path(__file__).read_text())

import traceback
import logging
logging.getLogger("clearml").setLevel(logging.DEBUG)

import os
import sys
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))
from clearml_scripts.utils import download_artifacts, upload_artifacts

sys.path.append(str(base_dir / "baselines" / "mt5"))
from pipeline import main

if __name__ == "__main__":
    try:
        download_artifacts()
        main()
        upload_artifacts()
    except:
        traceback.print_exc()
        raise
