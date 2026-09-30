import gymnasium as gym
import numpy as np
import torch
import robosuite

from resfit.dexmg.environments.dexmg import create_vectorized_env

def main():
    env = create_vectorized_env(
        env_name="Can",
        num_envs=4,
        camera_size=84,
    )
    # Don't seed explicitly
    obs, info = env.reset(seed=None)
    
    # Check if the states are identical
    state = obs["observation.state"]
    # state is of shape (4, state_dim)
    print("State 0:", state[0][:5])
    print("State 1:", state[1][:5])
    print("State 2:", state[2][:5])
    print("State 3:", state[3][:5])
    
    if torch.allclose(state[0], state[1]):
        print("WARNING: ENVIRONMENTS PRODUCED IDENTICAL STATES. They are not actually random!")
    else:
        print("SUCCESS: Environments are independent and different.")

if __name__ == "__main__":
    main()
