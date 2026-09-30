# 1. Import the listener at the top of your script
from trossen_real.human_intervention.pedal_listener import PedalListener
import time

# 2. Initialize it BEFORE your main loop starts
# This automatically spins up a background thread that listens to the pedal.
listener = PedalListener()

# ... inside your environment or training script ...
cntrl_freq = 20 # 20Hz
while True: # or inside env.step()
    start = time.time()
    # 3. Read the signals (these return instantly, no blocking)
    reward_signal = listener.get_reward()       # Returns 1 if held down, else 0
    intervention = listener.is_intervention()   # Returns True if held down, else False
    reset_env = listener.consume_reset()        # Returns True if it was pressed, and then resets itself
    
    if reset_env:
        print("Reset triggered!")
        continue
        
    if intervention:
        print("Intervention triggered!")
    
    if reward_signal:
        print("Reward triggered!")
        
    # Take the step
    # next_obs, env_reward, done, info = env.step(action)
    
    # Apply your custom reward signal from the pedal
    # if reward_signal == 1:
    #     env_reward = 1.0  # Or whatever you want to add
        
    # ... rest of your loop ...
    end = time.time()
    time.sleep(max(0, 1/cntrl_freq - (end - start)))