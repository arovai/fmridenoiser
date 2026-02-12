"""Setup configuration for fmridenoiser."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from package
version = {}
with open("fmridenoiser/core/version.py") as f:
    exec(f.read(), version)

# Read README
readme_path = Path("README.md")
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="fmridenoiser",
    version=version["__version__"],
    description="fMRI Denoising BIDS App for fMRIPrep Outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ln2t",
    url="https://github.com/ln2t/fmridenoiser",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "nibabel>=5.2.0",
        "nilearn>=0.10.3",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "pybids>=0.16.4",
        "PyYAML>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "colorama>=0.4.6",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "fmridenoiser=fmridenoiser.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
