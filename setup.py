import os
from setuptools import setup

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name="premise",
    version="0.1",
    author="Sebastian Junges",
    author_email="sebastian.junges@ru.nl",
    url="https://github.com/monitoring-MDPs/premise",
    description="Monitoring Systems with Imprecise Sensors",
    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=["premise"],
    install_requires=[
        "stormpy>=1.6.3", "tqdm", "pandas"
    ],
    python_requires='>=3',
)
