from setuptools import find_packages, setup


INSTALL_REQUIREMENTS = [
    "matplotlib==3.7.0",
    "numpy==1.23.4",
    "pandas==1.5.0",
    "PySimpleGUI==4.60.4",
    "scikit_gstat==1.0.9",
    "scikit_learn==1.2.1",
    "scipy==1.10.1",
]


SETUP = dict(
    name="iec62209",
    version="1.0.0",
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
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "iec62209-gui=iec62209.gui:main",
        ]
    },
    python_requires="~=3.10.8",
    install_requires=INSTALL_REQUIREMENTS,
)


if __name__ == "__main__":
    setup(**SETUP)
