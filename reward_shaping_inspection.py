import h5py
import numpy as np

GAMMA = 0.995
PENALTY = 0.01   # change this to test
SUCCESS_REWARD_VALUE = 1.0
SUCCESS_REWARD_MULTIPLIER = 10.0


def load_episodes(data):
    """Load the HDF5 data group into an in-memory episode list once."""
    episodes = []

    for ep_name in data.keys():
        ep_data = data[ep_name]
        episodes.append(
            {
                "name": ep_name,
                "rewards": ep_data["rewards"][:].astype(np.float64, copy=True),
                "dones": ep_data["dones"][:].astype(np.float64, copy=True),
            }
        )

    return episodes


def scale_success_rewards(episodes, multiplier, success_reward_value=1.0):
    """Return a copied episode list where success rewards are scaled."""
    scaled_episodes = []

    for episode in episodes:
        scaled_rewards = episode["rewards"].copy()
        success_mask = scaled_rewards == success_reward_value
        scaled_rewards[success_mask] *= multiplier

        scaled_episodes.append(
            {
                "name": episode["name"],
                "rewards": scaled_rewards,
                "dones": episode["dones"],
            }
        )

    return scaled_episodes

def compute_returns(rewards, gamma):
    """Compute discounted returns (like Q targets)"""
    returns = np.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def process_episode(ep_data):
    rewards = ep_data["rewards"]

    # original returns
    original_returns = compute_returns(rewards, GAMMA)

    # penalized rewards
    penalized_rewards = rewards - PENALTY
    penalized_returns = compute_returns(penalized_rewards, GAMMA)

    return {
        "len": len(rewards),
        "orig_start_Q": original_returns[0],
        "pen_start_Q": penalized_returns[0],
        "orig_total": rewards.sum(),
        "pen_total": penalized_rewards.sum(),
    }

def analyze_dataset(episodes):
    results = []

    for episode in episodes:
        res = process_episode(episode)
        results.append(res)

    # sort by episode length
    results.sort(key=lambda x: x["len"])

    print("\n=== SUMMARY ===")
    print(f"Num episodes: {len(results)}")

    print("\nShortest vs Longest comparison:\n")

    for r in [results[0], results[-1]]:
        print(f"Episode length: {r['len']}")
        print(f"  Original Q(start): {r['orig_start_Q']:.3f}")
        print(f"  Penalized Q(start): {r['pen_start_Q']:.3f}")
        print()

    # Correlation: length vs value
    lengths = np.array([r["len"] for r in results])
    q_vals = np.array([r["pen_start_Q"] for r in results])

    corr = np.corrcoef(lengths, q_vals)[0, 1]
    print(f"Correlation (length vs Q): {corr:.3f}")


# ============================================================================
# TEST A: Success Frequency (is signal too sparse?)
# ============================================================================

def test_success_frequency(episodes):
    """
    Measure success rate and density of success transitions.
    
    Interpretation:
    - < 10% success episodes: critic barely sees success ❌
    - 10–40%: healthy RL signal ✓
    - > 50%: easy task / strong signal ✓
    """
    total_steps = 0
    success_steps = 0
    episode_success = 0
    num_episodes = len(episodes)

    for episode in episodes:
        rewards = episode["rewards"]
        dones = episode["dones"]

        total_steps += len(rewards)

        # success = terminal step with done=1
        if np.any(dones == 1):
            episode_success += 1

        success_steps += np.sum(dones == 1)

    success_rate = episode_success / num_episodes
    success_density = success_steps / total_steps

    print("\n" + "="*70)
    print("TEST A: SUCCESS FREQUENCY")
    print("="*70)
    print(f"Total episodes: {num_episodes}")
    print(f"Success episodes: {episode_success}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Success step density: {success_density:.4f}")
    print()
    if success_rate < 0.1:
        print("⚠️  WARNING: < 10% success → critic may ignore success signal")
    elif success_rate < 0.4:
        print("✓ HEALTHY: 10-40% success rate")
    else:
        print("✓ STRONG: > 40% success rate")

    return {
        "success_rate": success_rate,
        "success_density": success_density,
        "num_episodes": num_episodes,
    }


# ============================================================================
# TEST B: Reward Mass Contribution (MOST IMPORTANT)
# ============================================================================

def test_success_reward_mass(episodes):
    """
    Measure contribution of success transitions to total discounted return.
    
    Interpretation:
    - < 0.1: success barely matters ❌
    - 0.1–0.4: moderate ⚠️
    - > 0.5: success dominates learning ✓
    """
    success_rewards = []
    total_rewards = []
    success_episode_count = 0

    for episode in episodes:
        rewards = episode["rewards"]
        dones = episode["dones"]

        total_rewards.append(np.sum(rewards))

        # reward at success steps only
        success_mask = (dones == 1)
        success_reward = np.sum(rewards[success_mask])
        success_rewards.append(success_reward)
        
        if np.any(success_mask):
            success_episode_count += 1

    success_rewards = np.array(success_rewards)
    total_rewards = np.array(total_rewards)

    # filter to episodes with success
    success_rewards_filtered = success_rewards[success_rewards > 0]
    
    if len(success_rewards_filtered) == 0:
        print("\n" + "="*70)
        print("TEST B: REWARD MASS CONTRIBUTION")
        print("="*70)
        print("⚠️  NO SUCCESS TRANSITIONS FOUND")
        print("Critic has no success signal to learn from!")
        return None

    avg_success = np.mean(success_rewards_filtered)
    avg_total = np.mean(total_rewards)
    ratio = avg_success / avg_total if avg_total > 0 else 0

    print("\n" + "="*70)
    print("TEST B: REWARD MASS CONTRIBUTION")
    print("="*70)
    print(f"Episodes with success: {success_episode_count}/{len(episodes)}")
    print(f"Avg reward at success steps: {avg_success:.4f}")
    print(f"Avg total episode reward: {avg_total:.4f}")
    print(f"Success/Total ratio: {ratio:.3f}")
    print()
    if ratio < 0.1:
        print("❌ PROBLEM: Success barely contributes to return")
    elif ratio < 0.4:
        print("⚠️  MODERATE: Success contributes but not dominant")
    else:
        print("✓ STRONG: Success dominates learning signal")

    return {
        "avg_success_reward": avg_success,
        "avg_total_reward": avg_total,
        "ratio": ratio,
        "success_episodes": success_episode_count,
    }


# ============================================================================
# TEST C: TD-Error Dominance (REAL critic signal test)
# ============================================================================

def td_error_proxy(rewards, gamma=0.995):
    """
    Approximate magnitude of critic TD targets (bootstrapped return).
    
    Higher magnitude = stronger learning signal.
    """
    G = 0
    td_signals = []

    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        td_signals.append(abs(G))

    return np.array(td_signals[::-1])  # reverse to match forward time


def test_td_error_dominance(episodes, gamma=0.995):
    """
    Compare TD-error magnitudes for success vs normal transitions.
    
    Interpretation:
    - < 1: success is weaker than normal transitions ❌
    - ~1: neutral signal ⚠️
    - > 1: success dominates learning ✓
    """
    success_signal = []
    normal_signal = []
    success_count = 0
    normal_count = 0

    for episode in episodes:
        rewards = episode["rewards"]
        dones = episode["dones"]

        td = td_error_proxy(rewards, gamma=gamma)

        for t in range(len(rewards)):
            if dones[t] == 1:
                success_signal.append(td[t])
                success_count += 1
            else:
                normal_signal.append(td[t])
                normal_count += 1

    success_signal = np.array(success_signal)
    normal_signal = np.array(normal_signal)

    if len(success_signal) == 0:
        print("\n" + "="*70)
        print("TEST C: TD-ERROR DOMINANCE")
        print("="*70)
        print("⚠️  NO SUCCESS TRANSITIONS FOUND")
        return None

    avg_success_td = np.mean(success_signal)
    avg_normal_td = np.mean(normal_signal)
    ratio = avg_success_td / avg_normal_td if avg_normal_td > 0 else 0

    # percentile analysis
    p50_success = np.percentile(success_signal, 50)
    p95_success = np.percentile(success_signal, 95)
    p50_normal = np.percentile(normal_signal, 50)
    p95_normal = np.percentile(normal_signal, 95)

    print("\n" + "="*70)
    print("TEST C: TD-ERROR DOMINANCE")
    print("="*70)
    print(f"Success transitions: {success_count}")
    print(f"Normal transitions: {normal_count}")
    print()
    print(f"Success TD magnitude (mean):  {avg_success_td:.6f}")
    print(f"Normal TD magnitude (mean):   {avg_normal_td:.6f}")
    print(f"Ratio (success/normal):       {ratio:.3f}")
    print()
    print(f"Success TD (median/p95):      {p50_success:.6f} / {p95_success:.6f}")
    print(f"Normal TD (median/p95):       {p50_normal:.6f} / {p95_normal:.6f}")
    print()
    if ratio < 1.0:
        print("❌ PROBLEM: Success is weaker than normal transitions")
        print("   Policy may ignore success in favor of plateau hovering")
    elif ratio < 1.5:
        print("⚠️  CAUTION: Success signal weak relative to normal trajectory")
    else:
        print("✓ STRONG: Success dominates TD learning signal")

    return {
        "success_count": success_count,
        "normal_count": normal_count,
        "avg_success_td": avg_success_td,
        "avg_normal_td": avg_normal_td,
        "ratio": ratio,
        "p50_success": p50_success,
        "p95_success": p95_success,
        "p50_normal": p50_normal,
        "p95_normal": p95_normal,
    }


# ============================================================================
# DIAGNOSTIC SUITE: Run all tests
# ============================================================================

def run_all_diagnostics(episodes, path, label):
    """Run all three success-dominance tests and summarize findings."""
    print("\n" + "="*70)
    print("RUNNING SUCCESS DOMINANCE DIAGNOSTIC SUITE")
    print("="*70)
    print(f"Dataset: {path}")
    print(f"Reward mode: {label}")

    results_a = test_success_frequency(episodes)
    results_b = test_success_reward_mass(episodes)
    results_c = test_td_error_dominance(episodes)

    print("\n" + "="*70)
    print("SUMMARY & INTERPRETATION")
    print("="*70)
    print()
    print("SAFE learning regime (success dominates):")
    print("  ✓ success rate: > 10%")
    print("  ✓ success reward ratio: > 0.2")
    print("  ✓ TD signal ratio: ≥ 1")
    print()
    print("BAD regime (policy may ignore success):")
    print("  ❌ success rare OR weak signal")
    print("  ❌ plateau rewards similar to success reward")
    print("  ❌ TD ratio < 1")
    print()

    # Quick assessment
    if results_a and results_b and results_c:
        safe = (
            results_a["success_rate"] > 0.1 and
            (results_b["ratio"] > 0.2 if results_b else False) and
            results_c["ratio"] >= 1.0
        )
        if safe:
            print("✓ OVERALL: Appears safe for reward shaping learning")
        else:
            print("❌ OVERALL: May have success signal problems")
    print()


if __name__ == "__main__":
    # Configuration: set your HDF5 path here
    FILE_PATH = "/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/Robomimic/robomimic/datasets/can/mh/robomimic-mh-can-image_v15_dense_filtered.hdf5"

    with h5py.File(FILE_PATH, "r") as dataset_file:
        data = dataset_file["data"]
        episodes = load_episodes(data)

    scaled_episodes = scale_success_rewards(
        episodes,
        multiplier=SUCCESS_REWARD_MULTIPLIER,
        success_reward_value=SUCCESS_REWARD_VALUE,
    )

    reward_label = (
        f"success rewards x{SUCCESS_REWARD_MULTIPLIER:.2f} when reward == {SUCCESS_REWARD_VALUE}"
    )

    # Run original analysis
    analyze_dataset(scaled_episodes)

    # Run success dominance tests
    run_all_diagnostics(scaled_episodes, FILE_PATH, reward_label)