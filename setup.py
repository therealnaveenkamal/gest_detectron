from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='gest_detectron',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    description='A package for gesture detection using Google - Hand Landmark Detection and Segmentation using SAM2.',
    author='Naveenraj Kamalakannan',
    author_email='naveenraj.k@nyu.edu',
    url='https://github.com/gest_detectron',
    entry_points={
        'console_scripts': [
            'gest-detectron=gest_detectron.main:main',
        ],
    },
)