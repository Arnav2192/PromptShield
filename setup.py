from setuptools import setup, find_packages

setup(
    name="promptshield",
    version="0.1.0",
    description="Hybrid, stateful firewall for LLM security",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "transformers>=4.30.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
