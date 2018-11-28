from setuptools import setup


setup(
    name='pytorch-template',
    version='0.0.1',
    packages=[
        'pytorch_template',
        'pytorch_template.models',
        'pytorch_template.scripts',
        'pytorch_template.utils',
    ],
    install_requires=[
        'gitpython',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'tensorboardX'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)