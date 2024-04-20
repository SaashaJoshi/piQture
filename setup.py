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

from setuptools import setup

# Read the contents of requirements.txt
with open("requirements.txt", "r") as reqs_file:
    install_requires = reqs_file.read().splitlines()

# Version
with open("piqture/version.txt", "r") as version_file:
    version = version_file.read().strip()

setup(
    name="piqture",
    version=version,
    description="piQture: A QML library for Image Processing",
    # long_description=README,
    author="Saasha Joshi",
    author_email="saashajoshi08@gmail.com",
    url="https://github.com/SaashaJoshi/piQture",
    packages=["piqture", "tests"],
    python_requires=">=3.6",
    install_requires=install_requires,
)
