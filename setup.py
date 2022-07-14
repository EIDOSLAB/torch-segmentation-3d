import sys
from os import path

from setuptools import setup, find_packages

if sys.version_info <= (3, 7):
    error = """Python {py} detected.
    torch-segmentation-3d supports only Python 3.8 and above.""".format(
        py=".".join([str(v) for v in sys.version_info[:3]])
    )

    sys.stderr.write(error + "\n")
    sys.exit(1)

here = path.abspath(path.dirname(__file__))

# Get the long description from the README.md file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def parse_requirements_file(filename):
    with open(filename, encoding="utf-8") as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


install_requires = parse_requirements_file("requirements.txt")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="torch-segmentation-3d",  # Required
    version="0.0.1-dev",  # Required
    description="3D segmentation models for pytorch",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/carloalbertobarbano/torch-segmentation-3d",  # Optional
    # This should be your name or the name of the organization which owns the
    # project.
    # author='A. Random Developer',  # Optional
    # This should be a valid email address corresponding to the author listed
    # above.
    # author_email='author@example.com',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(exclude=["tests"]),  # Required
    python_requires=">=3.5, <4",
)
