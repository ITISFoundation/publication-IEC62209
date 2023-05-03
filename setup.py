import re
import sys
from pathlib import Path

from setuptools import find_packages, setup


def read_reqs(reqs_path: Path) -> set[str]:
    return {
        r
        for r in re.findall(
            r"(^[^#\n-][\w\[,\]]+[-~>=<.\w]*)",
            reqs_path.read_text(),
            re.MULTILINE,
        )
        if isinstance(r, str)
    }


CURRENT_DIR = Path(sys.argv[0] if __name__ == "__main__" else __file__).resolve().parent

INSTALL_REQUIREMENTS = list( read_reqs(CURRENT_DIR / "requirements.txt") )


SETUP = dict(
    name="iec62209",
    version="1.0.6",
    description="Publication-IEC62209 package",
    author=", ".join(
        (
            "Cedric Bujard",
        )
    ),
    packages=find_packages(where="src"),
    package_dir={
        "": "src",
    },
    url="https://github.com/ITISFoundation/publication-IEC62209",
    license=Path("./LICENSE").read_text(),
    entry_points={
        "console_scripts": [
            "iec62209-gui=iec62209.gui:main",
        ]
    },
    python_requires="~=3.10.0",
    install_requires=INSTALL_REQUIREMENTS,
)


if __name__ == "__main__":
    setup(**SETUP)
