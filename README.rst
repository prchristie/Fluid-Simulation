# Fluid Simulation
This repository is a implementation of a simple fluid simulation as outlined from the paper
'Real-Time Fluid Dynamics for Games', adapted to work on cuda gpus. This will not work without
one.

## Requirements
To run this you will need a cuda capable gpu with CUDA development tools installed that allows it
to be utilized.

You will also need python3.6+ with venv installed to automatically make the virtual environment
for this repository

Any python requirements can be found in the requirements.txt and requirement_dev.txt, but it is
highly recommended that you follow the steps in Quickstart

## Quickstart
To make the environment necessary and connect to it;

```bash
make venv
source ./activate
```

and now to run the fluid simulation

```bash
fluid-sim
```

This is equivalent to running the main() function found in main.py