"""
Reward Functions
================
The reward function defines what "good driving" means.

At every timestep, the agent receives a number:
    positive → "this was a good action, do more of this"
    negative → "this was bad, avoid doing this"

The agent's goal is to maximize total reward over an episode.
There's no hardcoded driving logic — the agent discovers the
right behavior purely by maximizing this reward signal.

Available reward types:
    ProgressReward → Measures how far the car moved forward along
                     the track centerline. The most intuitive reward:
                     drive forward = good, crash = bad.

    CTHReward      → Cross-Track + Heading. Rewards the car for
                     pointing along the track direction AND staying
                     near the centerline. Produces smoother driving
                     than ProgressReward because it penalizes angling
                     toward walls even before a crash happens.

    SpeedReward    → Rewards raw speed. Best used in combination
                     with penalties (collision, steering, wall proximity)
                     to prevent the agent from just flooring it into walls.

Each reward type can be combined with optional penalties:
    - collision_penalty: large negative reward when the car crashes
    - steering_change_penalty: penalizes jerky steering between steps
    - wall_proximity_penalty: gentle nudge away from walls
    - survival_reward: small positive reward for not crashing
    - lap_bonus: bonus for completing a full lap
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os


class RewardFunction(ABC):
    """Base class for all reward functions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collision_penalty = config.get("collision_penalty", -10.0)
        self.lap_bonus = config.get("lap_bonus", 10.0)
        self.survival_reward = config.get("survival_reward", 0.0)
        self.steering_change_penalty = config.get("steering_change_penalty", 0.0)
        self.wall_proximity_penalty = config.get("wall_proximity_penalty", 0.0)
        self.wall_proximity_threshold = config.get("wall_proximity_threshold", 0.5)
        self._progress = 0.0

    def reset(self, obs_dict: Dict, ego_idx: int):
        self._progress = 0.0
        self._reset_impl(obs_dict, ego_idx)

    def _reset_impl(self, obs_dict, ego_idx):
        pass

    def compute(self, obs_dict, ego_idx, action, prev_action,
                terminated, collision, lap_complete) -> float:
        reward = self._compute_impl(obs_dict, ego_idx, action)
        reward += self.survival_reward

        # Steering smoothness penalty
        if self.steering_change_penalty > 0:
            reward -= self.steering_change_penalty * abs(action[0] - prev_action[0])

        # Wall proximity penalty — penalizes getting close to walls
        # Uses minimum lidar reading as proxy for wall distance
        if self.wall_proximity_penalty > 0:
            scan = obs_dict["scans"][ego_idx]
            min_dist = float(np.min(scan))
            if min_dist < self.wall_proximity_threshold:
                # Linear penalty: 0 at threshold, full penalty at 0
                penalty = 1.0 - (min_dist / self.wall_proximity_threshold)
                reward -= self.wall_proximity_penalty * penalty

        if collision:
            reward = self.collision_penalty
        elif lap_complete:
            reward += self.lap_bonus
        return reward

    @abstractmethod
    def _compute_impl(self, obs_dict, ego_idx, action) -> float:
        pass

    def get_progress(self) -> float:
        return self._progress


class ProgressReward(RewardFunction):
    """
    Progress-based reward. Rewards forward movement along the raceline.
    This is the most robust reward for racing RL.
    """

    def __init__(self, config: Dict[str, Any], waypoints: np.ndarray):
        super().__init__(config)
        self.waypoints = waypoints[:, :2]
        self.progress_weight = config.get("progress_weight", 10.0)
        self.speed_weight = config.get("speed_weight", 0.1)

        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        self.cumulative_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
        self.total_length = self.cumulative_dist[-1]
        self.prev_progress_dist = 0.0

    def _reset_impl(self, obs_dict, ego_idx):
        x = float(obs_dict["poses_x"][ego_idx])
        y = float(obs_dict["poses_y"][ego_idx])
        self.prev_progress_dist = self._get_progress_dist(x, y)
        self._progress = 0.0

    def _compute_impl(self, obs_dict, ego_idx, action) -> float:
        x = float(obs_dict["poses_x"][ego_idx])
        y = float(obs_dict["poses_y"][ego_idx])
        vel = float(obs_dict["linear_vels_x"][ego_idx])

        current_dist = self._get_progress_dist(x, y)
        delta = current_dist - self.prev_progress_dist
        if delta < -self.total_length * 0.5:
            delta += self.total_length
        elif delta > self.total_length * 0.5:
            delta -= self.total_length

        self.prev_progress_dist = current_dist
        self._progress += max(0, delta) / self.total_length

        reward = self.progress_weight * (delta / self.total_length)
        reward += self.speed_weight * max(0, vel)
        return reward

    def _get_progress_dist(self, x, y):
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        return self.cumulative_dist[np.argmin(dists)]


class CTHReward(RewardFunction):
    """
    Cross-Track-Heading reward (Evans et al. 2021).
    r = β_heading * v * cos(θ_error) - β_cross * d_crosstrack
    """

    def __init__(self, config: Dict[str, Any], waypoints: np.ndarray):
        super().__init__(config)
        self.waypoints = waypoints[:, :2]
        self.heading_weight = config.get("heading_weight", 0.04)
        self.crosstrack_weight = config.get("crosstrack_weight", 0.004)

        diffs = np.diff(self.waypoints, axis=0)
        self.wp_headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        self.wp_headings = np.append(self.wp_headings, self.wp_headings[-1])

        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        self.cumulative_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
        self.total_length = self.cumulative_dist[-1]
        self.prev_progress_dist = 0.0

    def _reset_impl(self, obs_dict, ego_idx):
        x, y = float(obs_dict["poses_x"][ego_idx]), float(obs_dict["poses_y"][ego_idx])
        self._update_progress(x, y)

    def _compute_impl(self, obs_dict, ego_idx, action) -> float:
        x = float(obs_dict["poses_x"][ego_idx])
        y = float(obs_dict["poses_y"][ego_idx])
        theta = float(obs_dict["poses_theta"][ego_idx])
        vel = float(obs_dict["linear_vels_x"][ego_idx])

        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        closest = np.argmin(dists)
        crosstrack = dists[closest]
        heading_err = self._norm_angle(theta - self.wp_headings[closest])

        reward = self.heading_weight * vel * np.cos(heading_err) - self.crosstrack_weight * crosstrack
        self._update_progress(x, y)
        return reward

    def _update_progress(self, x, y):
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        current_dist = self.cumulative_dist[np.argmin(dists)]
        delta = current_dist - self.prev_progress_dist
        if delta < -self.total_length * 0.5:
            delta += self.total_length
        self._progress += max(0, delta) / self.total_length
        self.prev_progress_dist = current_dist

    @staticmethod
    def _norm_angle(a):
        while a > np.pi: a -= 2 * np.pi
        while a < -np.pi: a += 2 * np.pi
        return a


class SpeedReward(RewardFunction):
    """Simple speed-proportional reward."""

    def __init__(self, config: Dict[str, Any], waypoints: np.ndarray):
        super().__init__(config)
        self.speed_weight = config.get("speed_weight", 0.1)
        self.waypoints = waypoints[:, :2]
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        self.cumulative_dist = np.concatenate([[0], np.cumsum(seg_lengths)])
        self.total_length = self.cumulative_dist[-1]
        self.prev_progress_dist = 0.0

    def _reset_impl(self, obs_dict, ego_idx):
        x, y = float(obs_dict["poses_x"][ego_idx]), float(obs_dict["poses_y"][ego_idx])
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        self.prev_progress_dist = self.cumulative_dist[np.argmin(dists)]

    def _compute_impl(self, obs_dict, ego_idx, action) -> float:
        return self.speed_weight * max(0, float(obs_dict["linear_vels_x"][ego_idx]))


class customReward(RewardFunction):
    """
    A Hybrid Reward Function combining Progress, CTH, and Speed.
    Inherits collision, lap, and smoothness logic from RewardFunction.
    """

    def __init__(self, config: Dict[str, Any], waypoints: np.ndarray):
        super().__init__(config)
        
        # 1. Initialize sub-components
        self.progress_comp = ProgressReward(config, waypoints)
        self.cth_comp      = CTHReward(config, waypoints)
        self.speed_comp    = SpeedReward(config, waypoints)
        
        # 2. Define weights for the hybrid calculation
        # These can be passed in via your config dictionary
        self.w_progress = config.get("w_progress", 1.0)
        self.w_cth      = config.get("w_cth", 1.0)
        self.w_speed    = config.get("w_speed", 1.0)

    def _reset_impl(self, obs_dict: Dict, ego_idx: int):
        """Called at the start of every episode to sync state."""
        self.progress_comp.reset(obs_dict, ego_idx)
        self.cth_comp.reset(obs_dict, ego_idx)
        self.speed_comp.reset(obs_dict, ego_idx)
        self._progress = 0.0

    def _compute_impl(self, obs_dict: Dict, ego_idx: int, action: np.ndarray) -> float:
        """Calculates the weighted driving reward."""
        
        # A. Get the 'raw' rewards from the components
        # We call _compute_impl to get just the logic, not the base penalties
        r_prog  = self.progress_comp._compute_impl(obs_dict, ego_idx, action)
        r_cth   = self.cth_comp._compute_impl(obs_dict, ego_idx, action)
        r_speed = self.speed_comp._compute_impl(obs_dict, ego_idx, action)

        # B. Combine them using the weights
        # Logic: Reward = (w1 * Progress) + (w2 * CTH) + (w3 * Speed)
        hybrid_reward = (
            (self.w_progress * r_prog) + 
            (self.w_cth      * r_cth)  + 
            (self.w_speed    * r_speed)
        )

        # C. Update the master progress tracker for logging
        self._progress = self.progress_comp.get_progress()

        return hybrid_reward

# ============================================================
# Waypoint loading (file-based fallback)
# ============================================================

def load_waypoints(config: Dict[str, Any], map_path: str) -> Optional[np.ndarray]:
    """Load waypoints from CSV files. Used as fallback when Track object unavailable."""
    for path_candidate in [
        config.get("raceline_path"),
        map_path + "_raceline.csv",
        map_path + "_centerline.csv",
    ]:
        if path_candidate and os.path.exists(path_candidate):
            try:
                data = np.loadtxt(path_candidate, delimiter=",", skiprows=1)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                return data[:, :2] if data.shape[1] < 3 else data[:, :3]
            except Exception:
                continue

    print("[WARNING] No waypoint file found. Using auto-generated waypoints.")
    print(f"  Searched: {config.get('raceline_path')}, {map_path}_raceline.csv, {map_path}_centerline.csv")
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    return np.column_stack([5.0 * np.cos(t), 5.0 * np.sin(t)])


def make_reward_function(config: Dict[str, Any], map_path: str) -> RewardFunction:
    """Create a reward function (file-based waypoint loading). Legacy interface."""
    wp = load_waypoints(config, map_path)
    reward_type = config.get("type", "progress")
    if reward_type == "progress":
        return ProgressReward(config, wp)
    elif reward_type == "cth":
        return CTHReward(config, wp)
    elif reward_type == "speed":
        return SpeedReward(config, wp)
    return ProgressReward(config, wp)
