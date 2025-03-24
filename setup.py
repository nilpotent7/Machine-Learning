from setuptools import setup, find_packages

setup(
    name="models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    author="Nilpotent",
    description="A machine learning module containing different model templates.",
    url="https://github.com/nilpotent7/Machine-Learning",
)
