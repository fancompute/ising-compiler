import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ising-compiler",
    version="0.1.0",
    author="Ben Bartlett",
    author_email="benbartlett@stanford.edu",
    description="🍰 Compiling your code to an Ising Hamiltonian so you don't have to!",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fancompute/ising-compiler",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "pydot"
    ],
)
