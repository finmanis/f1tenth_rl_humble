"""
Pure Pursuit Controller
========================
A simple waypoint-following algorithm that steers toward the next
point on the track. It's not RL — it's a classical control algorithm.

How it works:
    1. Find the waypoint that's a fixed distance ahead of the car
       (the "lookahead point")
    2. Compute the steering angle that would drive the car toward
       that point (using the bicycle model geometry)
    3. Set the speed based on the target speed from the config

Pure pursuit is used in this framework for three things:
    - As the "expert" for collecting imitation learning demonstrations
    - As the opponent in multi-agent training (RL vs pure pursuit)
    - As a baseline to compare your RL policy against

It's not a great racer (it follows the centerline at constant speed,
no racing line optimization) but it's reliable and predictable.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class PurePursuitController:
    """
    Pure pursuit waypoint tracker.

    Parameters
    ----------
    waypoints : np.ndarray, shape (N, 2+)
        Waypoints [x, y, ...]. Column 2 = target velocity if present.
    config : dict, optional
        Expert/pure_pursuit config section.
    """

    def __init__(self, waypoints: np.ndarray, config: Optional[Dict] = None):
        config = config or {}
        pp = config.get("pure_pursuit", config)

        self.waypoints = waypoints[:, :2]
        self.n_waypoints = len(waypoints)

        # Velocity profile: use waypoint velocities if they contain real speed data
        # (Track object fills default velocity as 1.0, which we ignore)
        if waypoints.shape[1] >= 3 and np.any(waypoints[:, 2] > 1.01):
            self.velocities = waypoints[:, 2].copy()
        else:
            self.velocities = np.full(self.n_waypoints, pp.get("target_speed", 5.0))

        self.adaptive_lookahead = pp.get("adaptive_lookahead", True)
        self.lookahead_distance = pp.get("lookahead_distance", 1.0)
        self.lookahead_gain = pp.get("lookahead_gain", 0.4)
        self.min_lookahead = pp.get("min_lookahead", 0.5)
        self.max_lookahead = pp.get("max_lookahead", 2.0)
        self.target_speed = pp.get("target_speed", 5.0)
        self.wheelbase = 0.33
        self.max_steer = 0.4189

        # Speed range: read from action config if provided, else defaults
        # MUST match the wrapper's action config for correct normalization
        action_cfg = config.get("_action_config", {})
        self.max_speed = action_cfg.get("max_speed", pp.get("max_speed", 8.0))
        self.min_speed = action_cfg.get("min_speed", pp.get("min_speed", 0.5))

    def set_waypoints(self, waypoints: np.ndarray):
        """Swap the raceline without rebuilding the controller (called at episode reset)."""
        self.waypoints = waypoints[:, :2]
        self.n_waypoints = len(waypoints)
        if waypoints.shape[1] >= 3 and np.any(waypoints[:, 2] > 1.01):
            self.velocities = waypoints[:, 2].copy()
        else:
            self.velocities = np.full(self.n_waypoints, self.target_speed)

    @classmethod
    def from_track(cls, track, config: Optional[Dict] = None) -> "PurePursuitController":
        """Create from an f1tenth_gym Track object."""
        line = getattr(track, "raceline", None) or getattr(track, "centerline", None)
        if line is None:
            raise ValueError("Track has no raceline or centerline")
        wp = np.column_stack([
            np.array(line.xs), np.array(line.ys),
            np.array(line.vxs) if hasattr(line, "vxs") else np.ones(len(line.xs)) * 5.0,
        ])
        return cls(wp, config)

    def get_action(self, obs_dict: Dict, ego_idx: int = 0) -> Tuple[float, float]:
        """
        Compute steering and speed.

        Parameters
        ----------
        obs_dict : dict
            Legacy flat observation dict.
        ego_idx : int
            Agent index.

        Returns
        -------
        (steer, speed) : tuple of float
        """
        x = float(obs_dict["poses_x"][ego_idx])
        y = float(obs_dict["poses_y"][ego_idx])
        theta = float(obs_dict["poses_theta"][ego_idx])
        vel = float(obs_dict["linear_vels_x"][ego_idx])

        # Closest waypoint
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        closest_idx = np.argmin(dists)

        # Lookahead distance
        if self.adaptive_lookahead:
            L = np.clip(self.lookahead_gain * max(vel, 0.5), self.min_lookahead, self.max_lookahead)
        else:
            L = self.lookahead_distance

        # Goal point at lookahead distance
        goal_idx = self._find_goal(closest_idx, L)
        goal = self.waypoints[goal_idx]

        # Transform to vehicle frame
        dx, dy = goal[0] - x, goal[1] - y
        goal_x_veh = np.cos(theta) * dx + np.sin(theta) * dy
        goal_y_veh = -np.sin(theta) * dx + np.cos(theta) * dy
        L_actual = max(np.sqrt(goal_x_veh**2 + goal_y_veh**2), 0.1)

        # Pure pursuit steering law
        curvature = 2.0 * goal_y_veh / (L_actual ** 2)
        steer = np.clip(np.arctan(curvature * self.wheelbase), -self.max_steer, self.max_steer)

        # Speed from velocity profile, slow for turns
        speed = float(self.velocities[goal_idx])
        if abs(steer) > 0.1:
            speed *= max(0.5, 1.0 - abs(steer) / self.max_steer)
        speed = np.clip(speed, self.min_speed, self.max_speed)

        return float(steer), float(speed)

    def get_normalized_action(self, obs_dict: Dict, ego_idx: int = 0) -> np.ndarray:
        """Get action in normalized [-1, 1] space."""
        steer, speed = self.get_action(obs_dict, ego_idx)
        steer_norm = steer / self.max_steer
        speed_norm = 2.0 * (speed - self.min_speed) / (self.max_speed - self.min_speed) - 1.0
        return np.array([steer_norm, speed_norm], dtype=np.float32)

    def _find_goal(self, closest_idx: int, lookahead: float) -> int:
        accumulated = 0.0
        idx = closest_idx
        for _ in range(self.n_waypoints):
            next_idx = (idx + 1) % self.n_waypoints
            seg = np.sqrt(
                (self.waypoints[next_idx, 0] - self.waypoints[idx, 0])**2
                + (self.waypoints[next_idx, 1] - self.waypoints[idx, 1])**2
            )
            accumulated += seg
            idx = next_idx
            if accumulated >= lookahead:
                return idx
        return (closest_idx + 10) % self.n_waypoints

    def get_tracking_info(self, obs_dict: Dict, ego_idx: int = 0) -> Dict:
        """Diagnostic info for debugging."""
        x = float(obs_dict["poses_x"][ego_idx])
        y = float(obs_dict["poses_y"][ego_idx])
        theta = float(obs_dict["poses_theta"][ego_idx])
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        closest = np.argmin(dists)
        return {
            "closest_idx": closest,
            "crosstrack_error": float(dists[closest]),
            "progress": closest / self.n_waypoints,
        }
