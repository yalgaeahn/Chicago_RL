import gym_examples
from gymnasium.envs.registration import register

register(
     id="gym_examples/Quantum",
     entry_point="gym_examples.envs:QuantumEnv")





