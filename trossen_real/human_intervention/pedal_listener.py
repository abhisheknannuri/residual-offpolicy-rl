"""

Usage
-----
    # 1. Import the listener at the top of your script
    from trossen_real.human_intervention.pedal_listener import PedalListener

    # 2. Initialize it BEFORE your main loop starts
    # This automatically spins up a background thread that listens to the pedal.
    listener = PedalListener()

    # ... inside your environment or training script ...
    while True: # or inside env.step()
        
        # 3. Read the signals (these return instantly, no blocking)
        reward_signal = listener.get_reward()       # Returns 1 if held down, else 0
        intervention = listener.is_intervention()   # Returns True if held down, else False
        reset_env = listener.consume_reset()        # Returns True if it was pressed, and then resets itself
        
        if reset_env:
            obs = env.reset()
            continue
            
        if intervention:
            # Override policy action with teleop / leader robot action
            action = get_teleop_action()
        else:
            # Normal policy action
            action = policy(obs)
            
        # Take the step
        next_obs, env_reward, done, info = env.step(action)
        
        # Apply your custom reward signal from the pedal
        if reward_signal == 1:
            env_reward = 1.0  # Or whatever you want to add
            
        # ... rest of your loop ...


"""

import logging
import threading
import evdev
from evdev import ecodes

logger = logging.getLogger(__name__)

PEDAL_NAME_HINTS = ("footpedal", "foot pedal", "vec")

class PedalListener:
    """
    A non-blocking background listener for the VEC foot pedal.
    Instantiate this once at the start of your program, and then simply 
    poll its state from your control loop or env.step().
    """
    def __init__(self, device_path=None):
        self._dev = None
        
        # Current state of each button (True if currently held down, False otherwise)
        self.reward_pressed = False
        self.intervention_pressed = False
        self.reset_pressed = False
        
        # Edge-triggered flag for reset (useful for catching a press that was already released)
        self._reset_triggered = False

        # Latched flag for reward: set True by ANY press event since the last
        # `consume_reward_latched()` call, even if the button is released again
        # before that call happens. Needed because a training loop only samples
        # pedal state once per control tick (after the arm has already moved) -
        # a press-then-release that happens ENTIRELY within a single tick's
        # ~50-100ms window must still count as reward=1 for that transition,
        # not be silently missed because the level was back to 0 by the time we
        # checked. See docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md.
        self._reward_latch = False
        
        # Try to open the device
        if device_path:
            self._dev = evdev.InputDevice(device_path)
        else:
            self._dev = self._find_pedal()
            
        if self._dev is None:
            logger.warning("No VEC foot pedal found! PedalListener will run in dummy mode.")
            self._dummy_mode = True
        else:
            self._dummy_mode = False
            logger.info("PedalListener initialized on %s (%s)", self._dev.path, self._dev.name)
            
            # Start the background thread
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
        
    def _find_pedal(self):
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except Exception:
                continue
            name = (dev.name or "").lower()
            if any(hint in name for hint in PEDAL_NAME_HINTS):
                return dev
        return None

    def _listen_loop(self):
        """Background loop reading from evdev without blocking the main program."""
        import time
        while True:
            try:
                # If disconnected, attempt to reconnect
                if self._dev is None:
                    logger.warning("Pedal disconnected - attempting to reconnect...")
                    for _ in range(24): # Retry every 5s for 120s (2min)
                        self._dev = self._find_pedal()
                        if self._dev is not None:
                            logger.info("Pedal reconnected on %s", self._dev.path)
                            break
                        time.sleep(5)

                    if self._dev is None:
                        logger.error("Failed to reconnect to pedal after 2 minutes. Stopping thread.")
                        return

                for event in self._dev.read_loop():
                    if event.type == ecodes.EV_KEY:
                        # event.value: 1 for press, 0 for release, 2 for hold
                        is_pressed = (event.value == 1 or event.value == 2)
                        
                        if event.code == 256:  # BTN_0 / BTN_MISC (Reward)
                            self.reward_pressed = is_pressed
                            if is_pressed:
                                self._reward_latch = True
                        elif event.code == 257:  # BTN_1 (Intervention)
                            self.intervention_pressed = is_pressed
                        elif event.code == 258:  # BTN_2 (Reset)
                            self.reset_pressed = is_pressed
                            # For reset, we also set an edge-trigger flag on press (not release/hold)
                            if event.value == 1:
                                self._reset_triggered = True
            except Exception as e:
                # Device might have been disconnected or closed
                logger.warning("PedalListener device error: %s. Connection lost.", e)
                if self._dev:
                    try:
                        self._dev.close()
                    except:
                        pass
                self._dev = None

    # --- Helper methods for the control loop ---

    def get_reward(self):
        """Returns 1 if the reward button is CURRENTLY pressed, 0 otherwise.
        Level-only (no latch) - prefer `consume_reward_latched()` in a control
        loop that only samples once per tick, so a press-then-release inside
        one tick isn't missed."""
        return 1 if self.reward_pressed else 0

    def consume_reward_latched(self):
        """Returns True if the reward button was pressed at ANY point since the
        last call (even if released again before this call), then resets the
        latch to the button's CURRENT held level - so continuously holding the
        pedal across many consecutive calls keeps returning True every time
        (matches 'every transition while success is being marked gets reward
        1', not just the instant it was pressed or released)."""
        latched = self._reward_latch or self.reward_pressed
        self._reward_latch = self.reward_pressed
        return latched

    def is_intervention(self):
        """Returns True if the intervention button is currently held down."""
        return self.intervention_pressed
        
    def consume_reset(self):
        """
        Returns True if the reset button was pressed since the last time this 
        method was called, then resets the flag. 
        This is perfect for triggering env.reset() exactly once per press.
        """
        if self._reset_triggered:
            self._reset_triggered = False
            return True
        return False
        
    def is_healthy(self):
        """Returns False if the pedal is in dummy mode or has lost its device
        connection (e.g. unplugged mid-run and not yet reconnected) - all button
        states are STALE/frozen at their last known values in that case, which
        is dangerous to rely on silently for reward-marking/reset/intervention
        during training. Callers should log/alert loudly if this goes False."""
        return not self._dummy_mode and self._dev is not None

    def stop(self):
        """Optionally close the device cleanly."""
        if not self._dummy_mode and self._dev:
            try:
                self._dev.close()
            except:
                pass


# --- Example usage if you run this script directly ---
if __name__ == "__main__":
    import time
    print("Testing PedalListener. Press pedals to see the state...")
    listener = PedalListener()
    
    try:
        while True:
            reward = listener.get_reward()
            intervention = listener.is_intervention()
            reset = listener.consume_reset()
            
            if reward or intervention or reset:
                print(f"State -> Reward: {reward} | Intervention: {intervention} | Reset triggered: {reset}")
            
            # Simulate a 10 Hz control loop
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting.")
