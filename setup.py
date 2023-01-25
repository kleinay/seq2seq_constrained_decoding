import setuptools
import os.path
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("src/constrained_decoding/version.txt", "r") as f:
    version = f.read().strip()

setuptools.setup(
    name="constrained_decoding",
    version=version,
    author="Ayal Klein",
    author_email="ayal.s.klein@gmail.com",
    description="package for constrained text generation during decoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kleinay/seq2seq_constrained_decoding",
    packages=setuptools.find_packages('src'),
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob('src/*.py')],
    package_dir={'': 'src'},
    install_requires=[
        'transformers>=4.15.0',
        'torch'
    ],
    # package_data={
    #     "": ["src/constrained_decoding/qasrl/data/qasrl_slots.json"],
    # },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
