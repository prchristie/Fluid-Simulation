#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirement_file:
    requirements = requirement_file.read()
requirements = requirements.split("\n")

setup(
    author="Patrick Christie",
    author_email='patrick.christie.dev@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A python implementation for a gpu accelerated fluid simulation as described in the paper 'Real-Time Fluid Dynamics for Games'",
    entry_points={
        'console_scripts': [
            'fluid-sim=fluid_sim.cli:start',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='fluid_sim',
    url='https://github.com/prchristie/Fluid-Simulation',
    version='0.0.1',
    zip_safe=False,
)