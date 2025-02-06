from setuptools import setup, find_packages

setup(
    name='gest_detectron',
    version='2.1.0',
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "mediapipe",
        "numpy",
        "opencv_contrib_python",
        "opencv_python",
        "Pillow",
        "setuptools",
        "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "torch",
        "torchvision",
        "torchaudio"
    ],
    description='A package for gesture detection using DinoV2 and Segmentation using SAM2.',
    author='Naveenraj Kamalakannan',
    author_email='naveenraj.k@nyu.edu',
    url='https://github.com/therealnaveenkamal/gest_detectron',
    entry_points={
        'console_scripts': [
            'gest-detectron=gest_detectron.main:main',
        ],
    },
)