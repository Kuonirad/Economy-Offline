"""
Setup configuration for Optimizer Trust Engine Framework
Production-grade package configuration
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core requirements (minimal dependencies)
install_requires = [
    # No hard dependencies - all optional for maximum compatibility
]

# Optional dependencies for enhanced functionality
extras_require = {
    "full": [
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
    ],
    "dev": [
        "pytest>=6.0.0",
        "pytest-asyncio>=0.14.0",
        "pytest-cov>=2.10.0",
        "black>=20.8b1",
        "flake8>=3.8.0",
        "mypy>=0.790",
        "pylint>=2.6.0",
    ],
    "monitoring": [
        "prometheus-client>=0.8.0",
        "psutil>=5.7.0",
    ],
}

setup(
    name="optimizer-trust-engine",
    version="2.1.0",
    author="GPU Optimization Team",
    author_email="team@optimizer-trust-engine.dev",
    description="Production-grade GPU optimization framework with Byzantine fault-tolerant verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kuonirad/Economy-Offline",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ote-demo=optimizer_trust_engine.demo:main",
            "ote-validate=optimizer_trust_engine.validate:main",
            "ote-test=optimizer_trust_engine.test_integration:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/Kuonirad/Economy-Offline/issues",
        "Source": "https://github.com/Kuonirad/Economy-Offline",
        "Documentation": "https://optimizer-trust-engine.readthedocs.io/",
    },
)