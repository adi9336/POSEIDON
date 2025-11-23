from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="poseidon",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Argo Float Data Processing and Analysis Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adi9336/POSEIDON",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "openai>=0.28.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "sqlalchemy>=1.4.0",
        "pydantic>=1.8.0",
        "requests>=2.26.0",
        "scikit-learn>=1.0.0",
        "geopy>=2.2.0",
        "python-multipart>=0.0.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "poseidon=poseidon.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
