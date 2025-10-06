import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="subgrapher",
    version="1.0.0",
    author="Lucas Morin",
    author_email="lum@zurich.ibm.com",
    description="A Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/LUM/SubGrapher-IBM/",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    install_requires=[
        "numpy==1.24.3",
        "pandas",
        "torch",
        "matplotlib",
        "tqdm",
        "torchvision",
        "rdkit",
        "scipy",
        "seaborn",
        "opencv-python"
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
)
