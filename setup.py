"""Setup script for DR-QFormer package."""

from pathlib import Path
from setuptools import setup, find_packages

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
    ]

setup(
    name="dr-qformer",
    version="0.1.0",
    author="DR-QFormer Contributors",
    author_email="",
    description="Parameter-Efficient Middleware for Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gugupig/DR-QFormer",
    packages=find_packages(exclude=["tests", "scripts", "configs", "assets"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
        "eval": [
            "rouge-score>=0.1.2",
            "sacrebleu>=2.3.0",
        ],
        "retrieval": [
            "faiss-cpu>=1.7.4",
        ],
        "logging": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drqf-demo=scripts.demo_cli:demo",
            "drqf-train-e=train.task_e:main",
            "drqf-train-s=train.task_s:main",
            "drqf-train-c=train.task_c:main",
            "drqf-eval=eval.evaluate:main",
        ],
    },
)
