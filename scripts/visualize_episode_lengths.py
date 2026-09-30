import matplotlib.pyplot as plt
import numpy as np
import scipy.special

def progressive_reward_function(current_epi_length, frame_idx):
    max_reward = 1 / (current_epi_length - 1)  # Maximum reward for the given episode length
    reward = 1 / (current_epi_length - frame_idx)
    return reward / max_reward  # Normalize to make it progressive

def log_reward_function(current_epi_length, frame_idx):
    reward = progressive_reward_function(current_epi_length, frame_idx)
    return np.log(reward) if reward > 0 else np.nan

def neg_log_reward_function(current_epi_length, frame_idx):
    reward = progressive_reward_function(current_epi_length, frame_idx)
    return -np.log(reward) if reward > 0 else np.nan

def logit_reward_function(current_epi_length, frame_idx):
    reward = progressive_reward_function(current_epi_length, frame_idx)
    if 0 < reward < 1:
        return scipy.special.logit(reward)
    else:
        return np.nan  # Handle invalid values gracefully

def visualize_rewards():
    episode_lengths = [100, 75, 50]
    max_episode_length = max(episode_lengths)
    frame_indices = np.arange(1, max_episode_length, 1)

    plt.figure(figsize=(15, 15))

    # Subplot 1: Progressive Reward Function
    plt.subplot(4, 2, 1)
    for epi_length in episode_lengths:
        rewards = [progressive_reward_function(epi_length, frame_idx) for frame_idx in frame_indices]
        plt.plot(frame_indices, rewards, label=f"Episode Length: {epi_length}")
    plt.title("Progressive Reward Function")
    plt.xlabel("Frame Index")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    # Subplot 2: Log of Progressive Reward Function
    plt.subplot(4, 2, 2)
    for epi_length in episode_lengths:
        log_rewards = [log_reward_function(epi_length, frame_idx) for frame_idx in frame_indices]
        plt.plot(frame_indices, log_rewards, label=f"Episode Length: {epi_length}")
    plt.title("Log of Progressive Reward Function")
    plt.xlabel("Frame Index")
    plt.ylabel("Log(Reward)")
    plt.legend()
    plt.grid()

    # Subplot 3: Negative Log of Progressive Reward Function
    plt.subplot(4, 2, 3)
    for epi_length in episode_lengths:
        neg_log_rewards = [neg_log_reward_function(epi_length, frame_idx) for frame_idx in frame_indices]
        plt.plot(frame_indices, neg_log_rewards, label=f"Episode Length: {epi_length}")
    plt.title("Negative Log of Progressive Reward Function")
    plt.xlabel("Frame Index")
    plt.ylabel("-Log(Reward)")
    plt.legend()
    plt.grid()

    # Subplot 4: Logit of Progressive Reward Function
    plt.subplot(4, 2, 4)
    for epi_length in episode_lengths:
        logit_rewards = [logit_reward_function(epi_length, frame_idx) for frame_idx in frame_indices]
        plt.plot(frame_indices, logit_rewards, label=f"Episode Length: {epi_length}")
    plt.title("Logit of Progressive Reward Function")
    plt.xlabel("Frame Index")
    plt.ylabel("Logit(Reward)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_rewards()