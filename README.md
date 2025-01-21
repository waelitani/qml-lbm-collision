# Towards Quantum Machine Learning of Collision Operators

This code is to accompany the final chapter of my thesis. It facilitates the construction and training of quantum neural networks to act as lattice Boltzmann collision operators using the `PennyLane v0.37` and `PyTorch 2.4` libraries.

If you find this code useful, please cite:

**Itani, W. (2025). Towards a Quantum Algorithm for Lattice Boltzmann (QALB) Simulation with a Nonlinear Collision Term. New York University.**

## Requirements (requirements.txt)
A file detailing the full Python environment used during the development of this code is made available.

## Dataset Generation (corbetta.py, create_trainset.py)
We make use of the dataset generation framework developed by [Corbetta et al.](https://doi.org/10.1140/epje/s10189-023-00267-w) and published in the corresponding [repository](https://github.com/agabbana/learning_lbm_collision_operator).

## Circuit Ansatz Construction (modularLayer.py)
A specialized class is developed to simplify the construction of the ansatz similar to the ones we used in our work. Some of the functionality of this class could be perfomed more efficiently with the registers of wires introduced in [PennyLane v0.38](https://pennylane.ai/blog/2024/09/pennylane-release-0.38).

## Training Hyperparameters (function_library.py)
Assisting functions are defined to load the trainingset and transform it to the fixed binary encoding.
The loss function is defined by making use of the stencil from [Corbetta et al.](https://doi.org/10.1140/epje/s10189-023-00267-w).
The D8 transforms and their implementation into the quantum circuit are also defined.

## Training Run (distributedelastic.py, torchquickstart.py, utils.py)
The files defining the overall training process, training loop and necessary assisting functions are defined in the above files respectively. The training is designed to run on a CPU cluster, and is executed with `torchrun` on the file containing the `main` function with appropriate arguments, e.g.:
```
torchrun distributedelastic.py --ne 10 --se 1 --ns 128 --bs 32 --fl circuit_library/SEL-CRY-Inverse-SEL/
```
The arguments accepted could be found in the `main` function definition.

## Demonstration (maruthinh.py, simulation.py)
The code used to demonstrate the utility of the learnt collision operator in lid-driven cavity flow within a hybrid setup is adapted from Maruthi N. Hanumantharayappa's [implementation](https://github.com/maruthinh/d2q9_zero_for_loop) to make use of the quantum circuitry. The simulation parameters are defined in `simulation.py` which is the file to be run to carry out the simulation.
