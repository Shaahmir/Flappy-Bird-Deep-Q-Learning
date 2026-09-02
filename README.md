# Deep Q-Learning Flappy Bird Agent

A Deep Reinforcement Learning implementation of a **Double Dueling Deep Q-Network (DD-DQN)** for learning to play Flappy Bird using **PyTorch** and **flappy-bird-gymnasium**. The project combines several established improvements over the original Deep Q-Network to improve training stability, sample efficiency, and convergence in sparse-reward environments.

---

## Overview

This repository implements a Deep Q-Learning agent capable of learning to play Flappy Bird from environment interactions. Rather than using a standard DQN, the agent incorporates modern architectural and optimization techniques to reduce value overestimation, improve representation learning, and stabilize training.

The implementation includes:

* Dueling Deep Q-Network architecture
* Double DQN optimization
* Experience Replay
* Soft Target Network Updates (Polyak Averaging)

---

## Architecture

### Dueling Deep Q-Network

The network separates state-value estimation from action-advantage estimation. Instead of directly predicting action values, it learns two functions:

* State Value, \(V(s)\)
* Action Advantage, \(A(s,a)\)

which are combined to estimate the final Q-value. This decomposition allows the model to identify valuable states without requiring accurate estimates for every possible action.

### Double DQN

Standard DQN suffers from overestimation caused by using the same network for both action selection and evaluation. Double DQN addresses this issue by selecting actions with the online network while evaluating them using the target network, producing more reliable value estimates.

### Experience Replay

Training samples are stored in a replay buffer and randomly sampled during optimization. This breaks the temporal correlation between consecutive transitions, improves sample efficiency, and stabilizes gradient updates.

### Soft Target Updates

Instead of periodically copying parameters to the target network, the implementation performs incremental updates using Polyak Averaging:

$$
\theta_{target}
\leftarrow
\tau\theta_{online}
+
(1-\tau)\theta_{target}
$$

where \($\tau = 0.005$).

---

## Repository Structure

```text

├── agent.py               # Training loop and optimization logic
├── dqn.py                 # Standard and Dueling DQN architectures
├── experience_replay.py   # Replay buffer implementation
├── main.py                # Training entry point
├── config.yaml            # Hyperparameter configuration
├── runs/                  # Training logs and saved checkpoints
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shaahmir/flappy-bird-dqn.git
cd flappy-bird-dqn
```

Install the required dependencies:

```bash
pip install torch gymnasium flappy-bird-gymnasium numpy pyyaml
```

or, if a requirements file is provided,

```bash
pip install -r requirements.txt
```

---

## Training

To begin training, run

```bash
python main.py
```

The agent will interact with the environment, collect experience, optimize the online network, perform soft updates on the target network, and periodically save model checkpoints.

---

## Configuration

Training hyperparameters are defined in `config.yaml`.

Common parameters include:

```yaml
learning_rate:
gamma:
batch_size:
buffer_size:
tau:
epsilon_start:
epsilon_end:
epsilon_decay:
```

---

## Implemented Features

* Double Deep Q-Network (Double DQN)
* Dueling Network Architecture
* Experience Replay Buffer
* Soft Target Updates (Polyak Averaging)
* Epsilon-Greedy Exploration
* PyTorch Implementation
* Configurable Hyperparameters

---

## References

* Mnih, V., et al. *Human-level control through deep reinforcement learning.* Nature, 2015.

* Van Hasselt, H., Guez, A., & Silver, D. *Deep Reinforcement Learning with Double Q-learning.* AAAI, 2016.

* Wang, Z., et al. *Dueling Network Architectures for Deep Reinforcement Learning.* ICML, 2016.

---

## License

This project is released under the MIT License. See the `LICENSE` file for additional information.
