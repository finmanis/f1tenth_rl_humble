"""
F1TENTH Gymnasium Wrapper
=========================
This is the bridge between F1TENTH Gym and RL algorithms.

The F1TENTH Gym simulator returns observations as nested Python
dictionaries (one per agent) with keys like 'scans', 'poses_x',
'linear_vels_x', etc. RL algorithms (SB3, CleanRL, custom PPO)
expect a flat numpy array as input and a flat numpy array as output.

This wrapper handles that translation:
    - Takes the raw dict from F1TENTH Gym
    - Extracts what the agent needs (lidar, velocity, waypoints)
    - Normalizes everything to neural-network-friendly ranges
    - Packs it into a flat vector that the policy network can process
    - Takes the policy's [-1, 1] output and converts it back to
      physical steering angle and speed

It also handles:
    - Loading maps and auto-generating centerlines
    - Multi-agent setup (pure pursuit opponents, self-play)
    - Domain randomization wrapping
    - Action smoothing for real-car deployment

You shouldn't need to modify this file for normal use — everything
is controlled through the YAML config. But if you're curious about
how the environment works under the hood, this is the place to look.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple

from f1tenth_rl.envs.observations import ObservationBuilder
from f1tenth_rl.envs.rewards import ProgressReward, CTHReward, SpeedReward, CustomReward
from f1tenth_rl.envs.domain_randomization import DomainRandomizationWrapper


# ============================================================
# Observation conversion
# ============================================================

def _flatten_obs_to_legacy(obs_dict: Dict, ego_idx: int, num_agents: int) -> Dict:
    """
    Convert dev-humble nested dict obs into flat legacy format.

    Dev-humble: {"agent_0": {"scan": ..., "std_state": [x,y,steer,vel,yaw,yaw_rate,slip], ...}}
    Legacy:     {"scans": [...], "poses_x": [...], "linear_vels_x": [...], ...}
    """
    legacy = {
        "scans": [], "poses_x": [], "poses_y": [], "poses_theta": [],
        "linear_vels_x": [], "linear_vels_y": [], "ang_vels_z": [],
        "collisions": [], "lap_times": [], "lap_counts": [],
    }
    for i in range(num_agents):
        agent_obs = obs_dict.get(f"agent_{i}", {})
        scan = agent_obs.get("scan", np.zeros(1080, dtype=np.float32))
        legacy["scans"].append(scan)

        std = agent_obs.get("std_state", np.zeros(7, dtype=np.float32))
        legacy["poses_x"].append(float(std[0]))
        legacy["poses_y"].append(float(std[1]))
        legacy["poses_theta"].append(float(std[4]))
        vel, beta = float(std[3]), float(std[6])
        legacy["linear_vels_x"].append(vel * np.cos(beta))
        legacy["linear_vels_y"].append(vel * np.sin(beta))
        legacy["ang_vels_z"].append(float(std[5]))

        legacy["collisions"].append(float(agent_obs.get("collision", 0.0)))
        legacy["lap_times"].append(float(agent_obs.get("lap_time", 0.0)))
        legacy["lap_counts"].append(float(agent_obs.get("lap_count", 0.0)))
    return legacy


# ============================================================
# Map file resolution for custom maps
# ============================================================

def _resolve_map_files(map_source: str) -> str:
    """Ensure map files match dev-humble naming: <n>_map.yaml + <n>_centerline.csv"""
    if "/" not in map_source and "\\" not in map_source:
        return map_source  # Built-in track name

    p = Path(map_source)
    new_yaml = p.parent / f"{p.stem}_map.yaml"
    old_yaml = p.parent / f"{p.stem}.yaml"

    if new_yaml.exists():
        pass
    elif old_yaml.exists():
        shutil.copy2(str(old_yaml), str(new_yaml))
        print(f"  [Map] Created {new_yaml.name} from {old_yaml.name}")
    else:
        raise FileNotFoundError(f"Map YAML not found: {new_yaml} or {old_yaml}")

    centerline = p.parent / f"{p.stem}_centerline.csv"
    if not centerline.exists():
        print(f"  [Map] Generating centerline from map image...")
        _generate_centerline(new_yaml, centerline)

    return str(p)


def _generate_centerline(yaml_path: Path, output_path: Path):
    """Generate a centerline CSV from a map image via skeletonization."""
    import yaml as pyyaml
    from PIL import Image
    from scipy.ndimage import binary_erosion
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        np.savetxt(str(output_path), np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    delimiter=",", header="x_m,y_m", comments="")
        return

    with open(yaml_path) as f:
        meta = pyyaml.safe_load(f)
    resolution = meta.get("resolution", 0.05)
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    image_file = meta.get("image")
    if not image_file:
        np.savetxt(str(output_path), np.array([[0, 0], [1, 0]]),
                    delimiter=",", header="x_m,y_m", comments="")
        return

    img = np.array(Image.open(yaml_path.parent / Path(image_file).name).convert("L"))
    free = binary_erosion((img > 200).astype(np.uint8), iterations=3).astype(np.uint8)
    skeleton = skeletonize(free > 0)
    pts = np.argwhere(skeleton)
    if len(pts) < 10:
        np.savetxt(str(output_path), np.array([[0, 0], [1, 0]]),
                    delimiter=",", header="x_m,y_m", comments="")
        return

    wx = pts[:, 1] * resolution + origin[0]
    wy = (img.shape[0] - pts[:, 0]) * resolution + origin[1]
    world = np.column_stack([wx, wy])

    # Nearest-neighbor ordering
    ordered = [world[0]]
    remaining = set(range(1, len(world)))
    for _ in range(len(world) - 1):
        if not remaining:
            break
        cur = ordered[-1]
        best_d, best_i = float("inf"), -1
        for idx in remaining:
            d = np.sqrt((world[idx, 0] - cur[0])**2 + (world[idx, 1] - cur[1])**2)
            if d < best_d:
                best_d, best_i = d, idx
        ordered.append(world[best_i])
        remaining.remove(best_i)
    ordered = np.array(ordered)

    # Subsample to ~10cm
    result = [ordered[0]]
    acc = 0.0
    for i in range(1, len(ordered)):
        d = np.sqrt((ordered[i, 0] - ordered[i-1, 0])**2 + (ordered[i, 1] - ordered[i-1, 1])**2)
        acc += d
        if acc >= 0.1:
            result.append(ordered[i])
            acc = 0.0

    np.savetxt(str(output_path), np.array(result), delimiter=",", header="x_m,y_m", comments="")
    print(f"  [Map] Generated centerline: {len(result)} points -> {output_path.name}")


# ============================================================
# Waypoint extraction from Track object
# ============================================================

def _extract_waypoints_from_track(track) -> np.ndarray:
    """
    Extract waypoints with velocities from a dev-humble Track object.

    Returns shape (N, 3): [x, y, velocity]
    """
    # Prefer raceline (has optimized velocities), fallback to centerline
    line = getattr(track, "raceline", None) or getattr(track, "centerline", None)
    if line is None:
        return None

    xs = np.array(line.xs, dtype=np.float64)
    ys = np.array(line.ys, dtype=np.float64)
    vxs = np.array(line.vxs, dtype=np.float64) if hasattr(line, "vxs") else np.ones_like(xs) * 5.0

    return np.column_stack([xs, ys, vxs])


# ============================================================
# Main Wrapper
# ============================================================

class F1TenthWrapper(gym.Env):
    """
    Gymnasium wrapper for F1TENTH Gym (dev-humble).

    Automatically extracts waypoints from the Track object for reward
    computation and waypoint-based observations.
    """

    metadata = {"render_modes": ["human", "human_fast", "rgb_array", "unlimited"]}

    def __init__(
        self,
        config: Dict[str, Any],
        render_mode: Optional[str] = None,
        ego_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.ego_idx = ego_idx
        self.render_mode = render_mode

        env_cfg = config["env"]
        obs_cfg = config["observation"]
        act_cfg = config["action"]
        rew_cfg = config["reward"]

        self.num_agents = env_cfg.get("num_agents", 1)
        self.max_steps = env_cfg.get("max_steps", 3000)
        self.timestep = env_cfg.get("timestep", 0.01)

        self.action_type = act_cfg.get("type", "continuous")
        self.max_speed = act_cfg.get("max_speed", 8.0)
        self.min_speed = act_cfg.get("min_speed", 0.5)
        self.max_steer = act_cfg.get("max_steer", 0.4189)
        self.smoothing_alpha = act_cfg.get("smoothing_alpha", 1.0)
        self.steer_dead_zone = act_cfg.get("steer_dead_zone", 0.0)
        self.max_steer_rate = act_cfg.get("max_steer_rate", 0.0)  # rad/step, 0 = unlimited

        # ---- Create base environment ----
        self.base_env = self._create_base_env(env_cfg, render_mode)

        # ---- Extract waypoints from Track ----
        self.waypoints = self._get_waypoints(rew_cfg, env_cfg)

        # ---- Observation builder ----
        obs_cfg_copy = dict(obs_cfg)
        # Get raw beam count: prefer actual sim value, fallback to lidar config
        try:
            obs_cfg_copy["_actual_raw_beams"] = self.base_env.unwrapped.sim.scan_num_beams
        except Exception:
            lidar_cfg = config.get("lidar", {})
            obs_cfg_copy["_actual_raw_beams"] = lidar_cfg.get("raw_beams", 1080)
        self.obs_builder = ObservationBuilder(obs_cfg_copy, self.num_agents)
        if self.waypoints is not None:
            self.obs_builder.set_waypoints(self.waypoints[:, :2])
        self.observation_space = self.obs_builder.get_observation_space()

        # ---- Action space ----
        if self.action_type == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        elif self.action_type == "discrete":
            n_s = act_cfg.get("num_speed_bins", 5)
            n_st = act_cfg.get("num_steer_bins", 7)
            self.action_space = spaces.Discrete(n_s * n_st)
            self._build_discrete_actions(n_s, n_st)
        elif self.action_type == "residual":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.residual_steer_range = act_cfg.get("residual_steer_range", 0.15)
            self.residual_speed_range = act_cfg.get("residual_speed_range", 2.0)
            self.base_controller = None

        # ---- Reward function ----
        self.reward_fn = self._make_reward(rew_cfg)

        # ---- Opponent controller (for multi-agent) ----
        self.opponent_controller = None
        self.opponent_rl_policy = None
        self.opponent_mode = config.get("multi_agent", {}).get("opponent", "pure_pursuit")
        if self.num_agents > 1 and self.waypoints is not None:
            if self.opponent_mode == "pure_pursuit":
                expert_wp = self._load_expert_waypoints(config)
                from f1tenth_rl.experts.pure_pursuit import PurePursuitController
                expert_cfg = dict(config.get("expert", {}))
                expert_cfg["_action_config"] = config.get("action", {})
                self.opponent_controller = PurePursuitController(
                    expert_wp, expert_cfg
                )

        # ---- Spawn config ----
        self.spawn_cfg = config.get("spawn", {})
        self._overtake_bonus = config["reward"].get("overtake_bonus", 50.0)

        # ---- Precompute cumulative track distances for overtake detection ----
        if self.waypoints is not None:
            _wps = self.waypoints[:, :2]
            _diffs = np.diff(_wps, axis=0)
            _seg_lens = np.sqrt((_diffs ** 2).sum(axis=1))
            self._track_cum_dist = np.concatenate([[0], np.cumsum(_seg_lens)])
            self._track_total_len = float(self._track_cum_dist[-1])
        else:
            self._track_cum_dist = None
            self._track_total_len = 0.0

        # ---- Episode state ----
        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_obs_dict = None
        self._episode_start_dist = 0.0
        self._overtake_done = False

    def _get_waypoints(self, rew_cfg: Dict, env_cfg: Dict) -> Optional[np.ndarray]:
        """Extract waypoints: Track object first, then file fallback."""
        try:
            track = self.base_env.unwrapped.track
            wp = _extract_waypoints_from_track(track)
            if wp is not None and len(wp) > 10:
                return wp
        except Exception:
            pass

        # File-based fallback
        map_path = env_cfg["map_path"]
        from f1tenth_rl.envs.rewards import load_waypoints
        return load_waypoints(rew_cfg, map_path)

    def _load_expert_waypoints(self, config: Dict) -> np.ndarray:
        """
        Load waypoints for pure pursuit / expert controller.

        Priority:
            1. expert.waypoint_path (custom raceline file)
            2. env.waypoints (auto-extracted from Track)

        This lets students use an optimized raceline for pure pursuit
        while the reward function still uses the centerline.
        """
        expert_cfg = config.get("expert", {})
        wp_path = expert_cfg.get("waypoint_path", None)

        if wp_path and os.path.exists(wp_path):
            try:
                data = np.loadtxt(wp_path, delimiter=",", skiprows=1)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                wp = data[:, :3] if data.shape[1] >= 3 else np.column_stack([data[:, :2], np.ones(len(data)) * 5.0])
                print(f"  [Expert] Loaded custom raceline: {wp_path} ({len(wp)} waypoints)")
                return wp
            except Exception as e:
                print(f"  [Expert] Failed to load {wp_path}: {e}")

        return self.waypoints

    def _make_reward(self, rew_cfg: Dict):
        """Create reward function using extracted waypoints."""
        reward_type = rew_cfg.get("type", "progress")
        wp = self.waypoints
        if wp is None:
            # Dummy waypoints
            t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
            wp = np.column_stack([5.0 * np.cos(t), 5.0 * np.sin(t)])

        if reward_type == "progress":
            return ProgressReward(rew_cfg, wp)
        elif reward_type == "cth":
            return CTHReward(rew_cfg, wp)
        elif reward_type == "speed":
            return SpeedReward(rew_cfg, wp)
        elif reward_type == "custom":
            return CustomReward(rew_cfg, wp)
        else:
            return ProgressReward(rew_cfg, wp)

    def _get_track_dist(self, x: float, y: float) -> float:
        """Cumulative distance along track to the nearest waypoint."""
        wps = self.waypoints[:, :2]
        dists = np.sqrt((wps[:, 0] - x) ** 2 + (wps[:, 1] - y) ** 2)
        return float(self._track_cum_dist[np.argmin(dists)])

    def _compute_spawn_states(self) -> np.ndarray:
        """
        Return (num_agents, 7) initial state array for the ST dynamic model.
        ST state: [x, y, delta, v, psi, psi_dot, beta]
        Ego is placed at a random waypoint; opponent is placed a random distance ahead.
        Both start at initial_speed.
        """
        wps = self.waypoints[:, :2]
        n = len(wps)

        ego_wp = int(self.np_random.integers(0, n))

        offset_m = float(self.np_random.uniform(
            self.spawn_cfg.get("opponent_offset_min", 3.0),
            self.spawn_cfg.get("opponent_offset_max", 10.0),
        ))

        # Walk forward along waypoints until cumulative distance >= offset_m
        opp_wp, dist_acc = ego_wp, 0.0
        for _ in range(n):
            nxt = (opp_wp + 1) % n
            dist_acc += float(np.sqrt(((wps[nxt] - wps[opp_wp]) ** 2).sum()))
            opp_wp = nxt
            if dist_acc >= offset_m:
                break

        def heading(idx: int) -> float:
            nxt = (idx + 1) % n
            return float(np.arctan2(wps[nxt, 1] - wps[idx, 1], wps[nxt, 0] - wps[idx, 0]))

        v0 = float(self.spawn_cfg.get("initial_speed", 2.0))
        states = np.zeros((self.num_agents, 7), dtype=np.float64)
        states[self.ego_idx] = [wps[ego_wp, 0], wps[ego_wp, 1], 0.0, v0, heading(ego_wp), 0.0, 0.0]
        opp_idx = 1 - self.ego_idx
        states[opp_idx]      = [wps[opp_wp, 0], wps[opp_wp, 1], 0.0, v0, heading(opp_wp), 0.0, 0.0]
        return states

    def _check_overtake(self, flat_obs: Dict) -> bool:
        """True if ego has passed the opponent by at least 0.5 m along the track."""
        if self.num_agents < 2 or self._track_cum_dist is None or self._overtake_done:
            return False
        opp_idx = 1 - self.ego_idx
        ego_d = self._get_track_dist(
            float(flat_obs["poses_x"][self.ego_idx]), float(flat_obs["poses_y"][self.ego_idx])
        )
        opp_d = self._get_track_dist(
            float(flat_obs["poses_x"][opp_idx]), float(flat_obs["poses_y"][opp_idx])
        )
        total = self._track_total_len
        adj_ego = (ego_d - self._episode_start_dist) % total
        adj_opp = (opp_d - self._episode_start_dist) % total
        return adj_ego > adj_opp + 0.5

    def _create_base_env(self, env_cfg: Dict, render_mode: Optional[str]):
        """Create F1TENTH env using dev-humble EnvConfig."""
        import f1tenth_gym
        from f1tenth_gym.envs.env_config import (
            EnvConfig, SimulationConfig, ObservationConfig,
            ResetConfig, ControlConfig,
        )
        from f1tenth_gym.envs.observation import ObservationType
        from f1tenth_gym.envs.reset import ResetStrategy
        from f1tenth_gym.envs.integrators import IntegratorType
        from f1tenth_gym.envs.dynamic_models import DynamicModel
        from f1tenth_gym.envs.lidar import LiDARConfig

        integrator = {"rk4": IntegratorType.RK4, "euler": IntegratorType.EULER}.get(
            env_cfg.get("integrator", "rk4").lower(), IntegratorType.RK4
        )
        map_source = _resolve_map_files(env_cfg["map_path"])

        # Read lidar hardware config
        lidar_cfg = self.config.get("lidar", {})
        raw_beams = lidar_cfg.get("raw_beams", 1080)
        fov_deg = lidar_cfg.get("fov_deg", 270.0)
        range_max = lidar_cfg.get("range_max", 30.0)

        env_config = EnvConfig(
            seed=self.config["experiment"].get("seed", 12345),
            map_name=map_source,
            num_agents=self.num_agents,
            ego_index=self.ego_idx,
            control_config=ControlConfig(steer_delay_steps=0),
            simulation_config=SimulationConfig(
                timestep=self.timestep,
                integrator=integrator,
                dynamics_model=DynamicModel.ST,
                max_laps=env_cfg.get("num_laps", 1),
            ),
            observation_config=ObservationConfig(type=ObservationType.DIRECT),
            reset_config=ResetConfig(strategy=ResetStrategy.RL_GRID_STATIC),
            lidar_config=LiDARConfig(
                enabled=True,
                num_beams=raw_beams,
                angle_min=np.deg2rad(-fov_deg / 2.0),
                angle_max=np.deg2rad(fov_deg / 2.0),
                range_max=range_max,
            ),
            render_enabled=(render_mode is not None),
        )
        return gym.make("f1tenth_gym:f1tenth-v0", config=env_config, render_mode=render_mode)

    # ---- Action methods ----

    def _build_discrete_actions(self, n_speed, n_steer):
        speeds = np.linspace(self.min_speed, self.max_speed, n_speed)
        steers = np.linspace(-self.max_steer, self.max_steer, n_steer)
        self.discrete_actions = np.array([[st, s] for s in speeds for st in steers])

    def set_base_controller(self, controller):
        self.base_controller = controller

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized [-1, 1] actions to physical [steer, speed]."""
        if self.action_type == "continuous":
            steer = float(action[0]) * self.max_steer
            speed = (float(action[1]) + 1.0) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed
        elif self.action_type == "discrete":
            raw = self.discrete_actions[int(action)]
            steer, speed = float(raw[0]), float(raw[1])
        elif self.action_type == "residual":
            if self.base_controller is not None and self.prev_obs_dict is not None:
                base_steer, base_speed = self.base_controller.get_action(self.prev_obs_dict)
            else:
                base_steer, base_speed = 0.0, self.min_speed
            steer = base_steer + float(action[0]) * self.residual_steer_range
            speed = base_speed + float(action[1]) * self.residual_speed_range
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

        steer = np.clip(steer, -self.max_steer, self.max_steer)
        speed = np.clip(speed, self.min_speed, self.max_speed)

        raw = np.array([steer, speed], dtype=np.float32)
        smoothed = self.smoothing_alpha * raw + (1 - self.smoothing_alpha) * self.prev_action

        # Steering dead zone: ignore tiny changes to prevent oscillation
        if self.steer_dead_zone > 0:
            steer_delta = smoothed[0] - self.prev_action[0]
            if abs(steer_delta) < self.steer_dead_zone:
                smoothed[0] = self.prev_action[0]

        # Steering rate limit: cap how fast steering can change per step
        if self.max_steer_rate > 0:
            steer_delta = smoothed[0] - self.prev_action[0]
            if abs(steer_delta) > self.max_steer_rate:
                smoothed[0] = self.prev_action[0] + np.sign(steer_delta) * self.max_steer_rate

        self.prev_action = smoothed.copy()

        all_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        all_actions[self.ego_idx] = smoothed

        # Opponents: use RL policy, pure pursuit, or drive slowly
        for i in range(self.num_agents):
            if i != self.ego_idx:
                if self.opponent_rl_policy is not None and self.prev_obs_dict is not None:
                    opp_steer, opp_speed = self._get_rl_opponent_action(i)
                    all_actions[i] = [opp_steer, opp_speed]
                elif self.opponent_controller is not None and self.prev_obs_dict is not None:
                    opp_steer, opp_speed = self.opponent_controller.get_action(self.prev_obs_dict, ego_idx=i)
                    all_actions[i] = [opp_steer, opp_speed]
                else:
                    all_actions[i] = [0.0, 1.0]

        return all_actions

    # ---- Core gym.Env interface ----

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        reset_options = dict(options) if options else {}

        # Random spawn with initial velocity via full-state initialization.
        # Falls back to start_pose config if waypoints are unavailable.
        if self.waypoints is not None and self._track_cum_dist is not None \
                and "states" not in reset_options and "poses" not in reset_options:
            reset_options["states"] = self._compute_spawn_states()
        else:
            start_cfg = self.config["env"].get("start_pose", None)
            if start_cfg is not None and "poses" not in reset_options:
                reset_options["poses"] = np.array([start_cfg] * self.num_agents, dtype=np.float64)

        if reset_options:
            raw_obs, info = self.base_env.reset(seed=seed, options=reset_options)
        else:
            raw_obs, info = self.base_env.reset(seed=seed)

        flat_obs = _flatten_obs_to_legacy(raw_obs, self.ego_idx, self.num_agents)

        self.current_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_obs_dict = flat_obs
        self.obs_builder.reset()
        self.reward_fn.reset(flat_obs, self.ego_idx)

        # Record ego's starting track position for overtake detection
        self._overtake_done = False
        if self._track_cum_dist is not None:
            self._episode_start_dist = self._get_track_dist(
                float(flat_obs["poses_x"][self.ego_idx]),
                float(flat_obs["poses_y"][self.ego_idx]),
            )

        observation = self.obs_builder.build(flat_obs, self.ego_idx, self.prev_action)
        info["raw_obs"] = flat_obs
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Save previous action BEFORE _scale_action updates it
        prev_physical_action = self.prev_action.copy()

        physical_action = self._scale_action(action)
        raw_obs, base_reward, done, truncated_flag, info = self.base_env.step(physical_action)
        self.current_step += 1

        flat_obs = _flatten_obs_to_legacy(raw_obs, self.ego_idx, self.num_agents)
        self.prev_obs_dict = flat_obs

        ego_collision = bool(flat_obs["collisions"][self.ego_idx])
        ego_lap_count = int(flat_obs["lap_counts"][self.ego_idx])
        num_laps = self.config["env"].get("num_laps", 1)
        terminated = done
        truncated = self.current_step >= self.max_steps

        reward = self.reward_fn.compute(
            obs_dict=flat_obs, ego_idx=self.ego_idx,
            action=physical_action[self.ego_idx],
            prev_action=prev_physical_action,  # Use the ACTUAL previous action
            terminated=terminated, collision=ego_collision,
            lap_complete=(ego_lap_count >= num_laps),
        )

        # Overtake detection: terminate and award bonus when ego passes the opponent
        overtake = self._check_overtake(flat_obs)
        if overtake:
            self._overtake_done = True
            terminated = True
            reward += self._overtake_bonus

        observation = self.obs_builder.build(flat_obs, self.ego_idx, self.prev_action)
        info.update({
            "raw_obs": flat_obs,
            "ego_collision": ego_collision,
            "ego_speed": float(flat_obs["linear_vels_x"][self.ego_idx]),
            "ego_lap_count": ego_lap_count,
            "ego_lap_time": float(flat_obs["lap_times"][self.ego_idx]),
            "progress": self.reward_fn.get_progress(),
            "step": self.current_step,
            "physical_action": physical_action[self.ego_idx].copy(),
            "overtake": overtake,
        })
        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is not None:
            return self.base_env.render()

    def close(self):
        self.base_env.close()

    # ---- Self-play methods (callable via env_method) ----

    def set_opponent_from_model_dict(self, state_dict):
        """Load opponent policy from state dict. Called by SelfPlayCallback."""
        import torch
        from f1tenth_rl.agents.networks import get_policy_kwargs

        if self.opponent_rl_policy is None:
            # Create policy network matching ego architecture
            obs_dim = self.observation_space.shape[0]
            act_dim = self.action_space.shape[0]
            from stable_baselines3.common.policies import ActorCriticPolicy
            import gymnasium as gym
            self.opponent_rl_policy = ActorCriticPolicy(
                observation_space=self.observation_space,
                action_space=self.action_space,
                lr_schedule=lambda _: 0.0003,
            )
        try:
            self.opponent_rl_policy.load_state_dict(state_dict)
            self.opponent_rl_policy.eval()
        except Exception:
            pass

    def _get_rl_opponent_action(self, agent_idx: int):
        """Get action from frozen RL opponent policy."""
        import torch
        try:
            obs = self.obs_builder.build(self.prev_obs_dict, ego_idx=agent_idx,
                                          prev_action=np.zeros(2, dtype=np.float32))
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = self.opponent_rl_policy._predict(obs_t, deterministic=True)
            action = action.cpu().numpy().squeeze()
            steer = float(action[0]) * self.max_steer
            speed = (float(action[1]) + 1.0) * 0.5 * (self.max_speed - self.min_speed) + self.min_speed
            return steer, speed
        except Exception:
            return 0.0, 1.0


# ============================================================
# Factory functions
# ============================================================

def make_env(config, rank=0, seed=0, render_mode=None) -> Callable:
    def _init():
        env = F1TenthWrapper(config, render_mode=render_mode)

        dr_cfg = config.get("domain_randomization", {})
        # Apply DR if: enabled=true (legacy) OR mode is "fixed"/"curriculum"
        dr_mode = dr_cfg.get("mode", "fixed" if dr_cfg.get("enabled", False) else "off")
        if dr_mode in ("fixed", "curriculum"):
            # Pass total_timesteps and num_envs so DR wrapper can self-track curriculum
            dr_cfg_with_steps = dict(dr_cfg)
            total_steps = config.get("algorithm", {}).get("total_timesteps", 2_000_000)
            num_envs = config.get("env", {}).get("num_envs", 8)
            # Each subprocess sees total_steps / num_envs steps
            dr_cfg_with_steps["curriculum_total_steps"] = total_steps // num_envs
            env = DomainRandomizationWrapper(env, dr_cfg_with_steps)

        # Wrap with episode statistics for gymnasium.vector episode tracking
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env
    return _init


def make_vec_env(config, n_envs=None, seed=0, normalize=True):
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

    if n_envs is None:
        n_envs = config["env"].get("num_envs", 8)

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(config, rank=i, seed=seed) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(config, rank=0, seed=seed)])

    vec_env = VecMonitor(vec_env)

    if normalize:
        algo_type = config["algorithm"]["type"]
        algo_cfg = config["algorithm"].get(algo_type, {})
        gamma = algo_cfg.get("gamma", 0.99) if isinstance(algo_cfg, dict) else 0.99
        # norm_obs=False: observations are already manually normalized
        # (lidar/clip, vel/10, yaw/pi). VecNormalize on top creates
        # sim2real mismatch — the running mean/var from sim don't match
        # real sensor distributions, causing jerky control on hardware.
        # norm_reward=True: reward normalization helps training stability
        # and doesn't affect deployment.
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0, gamma=gamma)

    return vec_env
