"""Setup configuration for the piQture package.

This script uses setuptools to package and distribute the piQture library.
It reads configuration from various files to set up the package.
"""

from setuptools import find_packages, setup

# Read the contents of requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as reqs_file:
    install_requires = reqs_file.read().splitlines()

# Read the version from version.txt
with open("piqture/version.txt", "r", encoding="utf-8") as version_file:
    version = version_file.read().strip()

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

# Setup configuration
setup(
    name="piqture",
    version=version,
    license="Apache-2.0",
    description="piQture: A QML library for Image Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Saasha Joshi",
    author_email="saashajoshi08@gmail.com",
    url="https://github.com/SaashaJoshi/piQture",
    packages=find_packages(include=["piqture", "piqture.*", "tests"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
