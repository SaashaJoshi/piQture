# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""piQture setup file."""

# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""piQture setup file."""

from setuptools import setup, find_packages

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
    description="piQture: A QML library for Image Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Saasha Joshi",
    author_email="saashajoshi08@gmail.com",
    url="https://github.com/SaashaJoshi/piQture",
    packages=find_packages(include=["piqture", "piqture.*", "tests"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
