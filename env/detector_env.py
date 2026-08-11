"""
detector_env.py

Custom Gymnasium environment: DetectorFaultControlEnv

A portable radiation-detector instrument is running. Underlying physical
processes (detector gain, battery, temperature, signal quality, vibration)
evolve stochastically and can drift into fault regimes. The agent observes
a short window of fused multi-sensor state (optionally augmented with an
LSTM's fault-probability estimate) and must choose a control action each
step to keep the instrument accurate, powered, and safe.

STATE (observation), per step, is the fused sensor vector:
    [count_rate, gain, battery_voltage, board_temp_c, signal_quality,
     vibration, lstm_fault_prob, steps_since_recalibration (normalized)]

ACTIONS (discrete, 5):
    0 = NOMINAL_HOLD        keep current sampling rate / power mode
    1 = RECALIBRATE         reset gain drift, costs power + a short blind window
    2 = THROTTLE_SAMPLING   reduce sampling rate -> saves power, lowers count SNR
    3 = BOOST_POWER_MODE    raise power draw -> improves signal quality/cooling
    4 = FLAG_MAINTENANCE    stop normal operation, request human service (terminal-ish)

REWARD design (this is the reward-engineering piece the JD calls out):
    + accuracy_reward   : higher when count_rate estimate matches true activity
                           (i.e. instrument is trustworthy)
    - power_penalty     : continuous power draw cost, worse in BOOST mode
    - false_trip_penalty: penalize FLAG_MAINTENANCE when no real fault is present
    - missed_fault_penalty: large penalty if a real fault persists uncaught
    - recalibration_cost: small cost each time RECALIBRATE is used (prevents spam)
    + uptime_bonus      : small reward per step the instrument stays operational

EPISODE ends on: FLAG_MAINTENANCE action, catastrophic battery depletion,
or max_steps reached.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DetectorFaultControlEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    ACTIONS = {
        0: "NOMINAL_HOLD",
        1: "RECALIBRATE",
        2: "THROTTLE_SAMPLING",
        3: "BOOST_POWER_MODE",
        4: "FLAG_MAINTENANCE",
    }

    def __init__(self, max_steps: int = 300, seed: int | None = None,
                 fault_prob_provider=None):
        super().__init__()
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # fault_prob_provider(obs_window) -> float in [0,1]; lets us plug in
        # the trained LSTM later. Defaults to a cheap heuristic stand-in so
        # the env is fully runnable standalone during early RL development.
        self.fault_prob_provider = fault_prob_provider or self._heuristic_fault_prob

        self.action_space = spaces.Discrete(5)
        low = np.array([0, 0.5, 4.5, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([300, 2.0, 8.6, 90, 1.0, 5.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._recent_window = []  # short history for the heuristic/LSTM provider
        self._window_len = 10

    def _heuristic_fault_prob(self, window):
        """Stand-in for the LSTM during standalone env testing: crude z-score
        based anomaly heuristic over the recent sensor window."""
        if len(window) < 3:
            return 0.0
        arr = np.array(window)
        temp = arr[:, 3]
        sig = arr[:, 4]
        gain = arr[:, 1]
        score = 0.0
        score += max(0.0, (temp[-1] - 30) / 40)
        score += max(0.0, (1 - sig[-1]) - 0.1)
        score += max(0.0, abs(gain[-1] - 1.0) - 0.02) * 3
        return float(np.clip(score, 0.0, 1.0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.gain = 1.0
        self.battery = 8.4
        self.temp = 25.0 + self._rng.normal(0, 0.5)
        self.signal_quality = 0.95
        self.vibration = 0.05
        self.power_mode = "normal"     # normal | throttled | boosted
        self.steps_since_recal = 0
        self._recent_window = []

        # randomly decide if/when this episode develops a latent fault
        self.has_fault = self._rng.random() < 0.6
        self.fault_type = self._rng.choice([1, 2, 3, 4]) if self.has_fault else 0
        self.fault_onset = self._rng.integers(50, self.max_steps - 50) if self.has_fault else None
        self.fault_flagged = False

        obs = self._get_obs()
        return obs, {}

    def _step_physics(self, action_name: str):
        # baseline drift
        self.battery -= 0.0015
        if self.power_mode == "boosted":
            self.battery -= 0.003
        elif self.power_mode == "throttled":
            self.battery += 0.0008  # saves power

        self.temp += self._rng.normal(0.01, 0.05)
        if self.power_mode == "boosted":
            self.temp += 0.03  # boosting runs hotter

        # fault progression
        if self.has_fault and self.fault_onset is not None and self.t >= self.fault_onset:
            progress = min(1.0, (self.t - self.fault_onset) / 80.0)
            if self.fault_type == 1:  # gain drift
                self.gain = 1.0 + progress * 0.25
            elif self.fault_type == 2:  # power sag
                self.battery -= progress * 0.01
            elif self.fault_type == 3:  # thermal
                self.temp += progress * 0.15
                self.signal_quality -= progress * 0.003
            elif self.fault_type == 4:  # signal fault
                self.signal_quality -= progress * 0.01
                self.vibration = 0.05 + progress * self._rng.uniform(0.2, 0.6)

        # actions affect physics
        if action_name == "RECALIBRATE":
            self.gain = 1.0
            self.steps_since_recal = 0
        else:
            self.steps_since_recal += 1

        if action_name == "BOOST_POWER_MODE":
            self.power_mode = "boosted"
            self.signal_quality = min(1.0, self.signal_quality + 0.01)
        elif action_name == "THROTTLE_SAMPLING":
            self.power_mode = "throttled"
        else:
            self.power_mode = "normal"

        self.signal_quality = float(np.clip(
            self.signal_quality + self._rng.normal(0, 0.005), 0.0, 1.0))
        self.temp = float(np.clip(self.temp, 15, 90))
        self.battery = float(np.clip(self.battery, 0, 8.6))
        self.vibration = float(np.clip(self.vibration + self._rng.normal(0, 0.02), 0, 5))

        base_rate = 120.0
        thermal_bias = max(0.05, 1 - 0.01 * max(0, self.temp - 25))
        lam = max(1.0, base_rate * self.gain * thermal_bias * self.signal_quality)
        self.count_rate = float(self._rng.poisson(lam))

    def _get_obs(self):
        row = [self.count_rate if hasattr(self, "count_rate") else 120.0,
               self.gain, self.battery, self.temp, self.signal_quality,
               self.vibration]
        self._recent_window.append(row)
        self._recent_window = self._recent_window[-self._window_len:]
        fault_prob = self.fault_prob_provider(self._recent_window)
        obs = np.array(row + [fault_prob, min(1.0, self.steps_since_recal / 100)],
                        dtype=np.float32)
        return obs

    def step(self, action: int):
        action_name = self.ACTIONS[int(action)]
        self._step_physics(action_name)
        self.t += 1

        obs = self._get_obs()
        fault_prob = obs[6]

        # --- reward engineering ---
        true_fault_active = (self.has_fault and self.fault_onset is not None
                              and self.t >= self.fault_onset)

        accuracy_reward = 1.0 - abs(self.gain - 1.0) - (1.0 - self.signal_quality) * 0.5
        power_penalty = {"normal": 0.02, "throttled": 0.005, "boosted": 0.06}[self.power_mode]
        recal_cost = 0.3 if action_name == "RECALIBRATE" else 0.0
        uptime_bonus = 0.05

        terminated = False
        false_trip_penalty = 0.0
        missed_fault_penalty = 0.0

        if action_name == "FLAG_MAINTENANCE":
            self.fault_flagged = True
            terminated = True
            if not true_fault_active:
                false_trip_penalty = 5.0   # flagged with nothing wrong
            else:
                accuracy_reward += 3.0     # correctly caught a real fault

        if true_fault_active and not self.fault_flagged and fault_prob > 0.8:
            # agent "should" have acted more decisively; small shaping penalty
            missed_fault_penalty = 0.1

        battery_dead = self.battery <= 0.05
        if battery_dead:
            terminated = True
            missed_fault_penalty += 5.0

        reward = (accuracy_reward + uptime_bonus
                  - power_penalty - recal_cost
                  - false_trip_penalty - missed_fault_penalty)

        truncated = self.t >= self.max_steps
        info = {
            "true_fault_active": true_fault_active,
            "fault_type": self.fault_type,
            "power_mode": self.power_mode,
            "battery": self.battery,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(f"t={self.t:03d} gain={self.gain:.3f} batt={self.battery:.2f}V "
              f"temp={self.temp:.1f}C sig={self.signal_quality:.2f} "
              f"mode={self.power_mode}")


if __name__ == "__main__":
    env = DetectorFaultControlEnv(seed=0)
    obs, info = env.reset()
    total_r = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            break
    print("Random-policy smoke test OK. Steps:", env.t, "Total reward:", round(total_r, 2))
    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)
