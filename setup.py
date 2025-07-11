from setuptools import find_packages, setup

setup(
    name='stocksimpy',
    packages=find_packages(include=['stocksimpy']),
    version='0.1.0',
    description='A lightweight library for stock strategy testing.',
    author='Suleyman Sade',
    install_requires=[],
    setup_requires=['pytest-runner'],
    test_requires=['pytest']
)