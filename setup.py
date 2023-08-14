from setuptools import find_packages, setup

setup(
    name="rl-min-app",
    version="0.0.1",
    description="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "ray[rllib]",
        "black",
        "flake8",
        "ipykernel",
        "ipywidgets",
    ],
)
