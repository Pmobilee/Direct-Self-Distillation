from setuptools import setup, find_packages

setup(
    name='DSD',
    version='0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)