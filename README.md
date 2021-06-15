# Reinforcement Learning for Sequential Assembly of SL-Blocks

This repository contains Grasshopper files and Python scripts for running sequential assembly
experiments from the paper "Reinforcement Learning for Sequential Assembly of SL-Blocks".
Reinforcement learning library [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
is used for training the agents, while the environments are implemented
in [Grasshopper](https://www.grasshopper3d.com/) — an algorithmic modeling tool for Rhino.


## Installation

Required Rhino/Gasshopper plugins can be downloaded from [Food4Rhino](https://www.food4rhino.com/)
and installed as described on their respective pages
* Hoopsnake
* Matrixtoolbox
* Treesloth
* Yellow
* Zebra

For a general description of the interface between Grasshopper and Stable-Baselines3,
see [Grasshopper Gym](https://github.com/b4be1/gh_gym). It is recommended to use
[Anaconda](https://www.anaconda.com/products/individual#Downloads) to get a running Python
installation under Windows.


## Training and Evaluating Agents

The basic script `train-and-eval.py` provides the code for training and evaluating agents.
To run the training, execute the command
```
python train-and-eval.py --algorithm="PPO" --save-path="DESIRED_PATH"
```
Here you need to specify the algorithm (either PPO or DQN) and the path where the model should be
saved. Later, to evaluate the saved model, execute the command
```
python train-and-eval.py --algorithm="PPO" --load-path="DESIRED_PATH"
```
Before executing these commands, start the Hoopsnake loop inside the Grasshopper file
`Reinforcement Learning Interface.gh` located in the `Grasshopper Files` directory.
Parameters of the reinforcement learning algorithms can be adjusted directly
in the `train-and-eval.py` script where DQN and PPO objects are created (lines 92–102).


## Experiments on 2D and 3D Curves

Python scripts for training DQN and PPO agents on the 2D and 3D curves described in the paper
are provided in the folder `Python Files` with the parameters are already pre-specified.
Detailed instructions on how to use those scripts and how to modify the Grasshopper environment
in `Reinforcement Learning Interface.gh` can be found in the accompanying video
[Reinforcement Learning for Sequential Assembly of SL-Blocks in Grasshopper](https://youtu.be/owHATVWgNk4).
Other algorithms used as baselines in the paper are implemented directly in Grasshopper
in the file `Evolutionary Algorithm and Greedy Algorithm for SL-Blocks.gh`.
Pre-trained weights for DQN and PPO agents are available in the directories
`DQN-Experiments` and `PPO-Experiments`.
