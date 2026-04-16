#!/usr/bin/env python3
"""
F1TENTH RL Evaluation Script
==============================
Evaluate trained policies and analyze performance.

Usage:
    # Evaluate from a run directory
    python scripts/evaluate.py --run runs/ppo_Spielberg_2026-04-04_18-30-00

    # Evaluate a specific model file
    python scripts/evaluate.py --model checkpoints/my_model

    # Evaluate best model from run
    python scripts/evaluate.py --run runs/my_run --use-best

    # Record trajectory plots + animated GIF (no display needed, works on headless VMs)
    python scripts/evaluate.py --run runs/my_run --record --episodes 10

    # Save recordings to a specific directory
    python scripts/evaluate.py --run runs/my_run --record --record-dir /tmp/my_eval

    # Compare multiple runs
    python scripts/evaluate.py --run runs/run1 runs/run2 --plot

    # Export to ONNX
    python scripts/evaluate.py --run runs/my_run --export-onnx
"""

import argparse
import os
import sys
import yaml
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_model_and_config(run_or_model: str, use_best: bool = False):
    """
    Resolve a run directory or model path to (model_path, config_path, norm_path).

    Handles:
        runs/my_run/                -> final_model + config.yaml + final_vecnormalize.pkl
        runs/my_run/ --use-best     -> best_model/best_model + config.yaml
        checkpoints/my_model        -> my_model + my_model_config.yaml
    """
    p = Path(run_or_model)

    if p.is_dir():
        # It's a run directory
        config_path = str(p / "config.yaml")
        if use_best:
            model_path = str(p / "best_model" / "best_model")
            norm_path = str(p / "final_vecnormalize.pkl")  # best doesn't have its own
        else:
            model_path = str(p / "final_model")
            norm_path = str(p / "final_vecnormalize.pkl")
    else:
        # It's a model path
        model_path = str(p)
        config_path = str(p) + "_config.yaml"
        if not os.path.exists(config_path):
            config_path = str(p.parent / "config.yaml")
        norm_path = str(p) + "_vecnormalize.pkl"
        if not os.path.exists(norm_path):
            norm_path = str(p.parent / "final_vecnormalize.pkl")

    return model_path, config_path, norm_path


def evaluate_model(model_path, config, norm_path, args):
    """
    Evaluate a single model.

    Returns:
        (metrics, episodes_data, waypoints)
        episodes_data is a list of per-episode dicts when --record is set, else None.
        waypoints is the env's waypoint array (for trajectory plotting), else None.
    """
    from f1tenth_rl.envs.wrapper import F1TenthWrapper
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor

    algo_type = config["algorithm"]["type"]
    AlgoClass = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo_type]

    # Create single env
    render_mode = "human" if args.render else None
    env = F1TenthWrapper(config, render_mode=render_mode)
    ego_idx = env.ego_idx

    # Load model
    model = AlgoClass.load(model_path, device="cpu")

    # Load normalization stats if available
    obs_rms = None
    if os.path.exists(norm_path):
        import pickle
        try:
            with open(norm_path, "rb") as f:
                vec_norm = pickle.load(f)
            obs_rms = vec_norm.obs_rms
            print(f"  Loaded normalization stats from {norm_path}")
        except Exception:
            pass

    metrics = defaultdict(list)
    record = getattr(args, "record", False)
    episodes_data = [] if record else None

    print(f"\nEvaluating: {model_path}")
    print(f"  Algorithm: {algo_type.upper()}")
    print(f"  Episodes:  {args.episodes}")
    print("-" * 50)

    for ep in range(args.episodes):
        obs, info = env.reset()
        episode_return = 0
        speeds = []
        steers = []
        done = False
        frames = [] if record else None
        overtake_step = None
        step_count = 0

        while not done:
            # Normalize obs if we have stats
            obs_input = obs
            if obs_rms is not None:
                obs_input = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
                obs_input = np.clip(obs_input, -10.0, 10.0).astype(np.float32)

            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

            speeds.append(info.get("ego_speed", 0))
            steers.append(abs(info.get("physical_action", [0, 0])[0]))

            if record:
                raw = info.get("raw_obs", {})
                poses_x = raw.get("poses_x", [])
                poses_y = raw.get("poses_y", [])
                opp_idx = 1 - ego_idx
                frames.append({
                    "ego_x": float(poses_x[ego_idx]) if len(poses_x) > ego_idx else 0.0,
                    "ego_y": float(poses_y[ego_idx]) if len(poses_y) > ego_idx else 0.0,
                    "opp_x": float(poses_x[opp_idx]) if len(poses_x) > opp_idx else None,
                    "opp_y": float(poses_y[opp_idx]) if len(poses_y) > opp_idx else None,
                })
                if info.get("overtake") and overtake_step is None:
                    overtake_step = step_count

            if args.render:
                env.render()
                time.sleep(0.005)

            step_count += 1

        metrics["return"].append(episode_return)
        metrics["steps"].append(info.get("step", 0))
        metrics["avg_speed"].append(np.mean(speeds))
        metrics["max_speed"].append(np.max(speeds))
        metrics["progress"].append(info.get("progress", 0))
        metrics["collision"].append(info.get("ego_collision", False))
        metrics["lap_time"].append(info.get("ego_lap_time", 0))
        metrics["overtake"].append(info.get("overtake", False))
        if len(steers) > 1:
            metrics["steer_smoothness"].append(np.std(np.diff(steers)))

        outcome = (
            "OVERTAKE" if info.get("overtake")
            else "CRASH" if info.get("ego_collision")
            else "timeout"
        )
        print(
            f"  Episode {ep+1:3d}/{args.episodes} | "
            f"Return: {episode_return:8.1f} | "
            f"Progress: {info.get('progress', 0):6.2%} | "
            f"Speed: {np.mean(speeds):5.1f} m/s | "
            f"{outcome}"
        )

        if record:
            episodes_data.append({
                "frames": frames,
                "return": episode_return,
                "overtake": info.get("overtake", False),
                "collision": info.get("ego_collision", False),
                "overtake_step": overtake_step,
                "steps": step_count,
            })

    waypoints = getattr(env, "waypoints", None)
    env.close()
    return metrics, episodes_data, waypoints


def print_results(metrics, name="Model"):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"  Results: {name}")
    print(f"{'='*60}")
    for key, values in metrics.items():
        v = np.array(values)
        if key == "collision":
            print(f"  {key:20s}: {np.mean(v):.1%} crash rate ({np.sum(v):.0f}/{len(v)} episodes)")
        elif key == "overtake":
            print(f"  {key:20s}: {np.mean(v):.1%} success rate ({np.sum(v):.0f}/{len(v)} episodes)")
        elif key == "progress":
            print(f"  {key:20s}: {np.mean(v):.2%} ± {np.std(v):.2%}  (max: {np.max(v):.2%})")
        elif key == "lap_time":
            non_zero = v[v > 0]
            if len(non_zero) > 0:
                print(f"  {key:20s}: {np.mean(non_zero):.2f}s  (best: {np.min(non_zero):.2f}s)")
            else:
                print(f"  {key:20s}: no laps completed")
        else:
            print(f"  {key:20s}: {np.mean(v):.3f} ± {np.std(v):.3f}")
    print(f"{'='*60}\n")


def plot_results(all_metrics, names, save_path=None):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("F1TENTH RL Policy Evaluation", fontsize=14)

    for name, metrics in zip(names, all_metrics):
        axes[0, 0].plot(metrics["return"], alpha=0.5, label=name)
        axes[0, 0].set_title("Episode Returns")
        axes[0, 0].legend()

        axes[0, 1].hist(metrics["progress"], bins=20, alpha=0.5, label=name)
        axes[0, 1].set_title("Track Progress Distribution")
        axes[0, 1].legend()

        axes[1, 0].hist(metrics["avg_speed"], bins=20, alpha=0.5, label=name)
        axes[1, 0].set_title("Average Speed (m/s)")
        axes[1, 0].legend()

        if "steer_smoothness" in metrics:
            axes[1, 1].hist(metrics["steer_smoothness"], bins=20, alpha=0.5, label=name)
            axes[1, 1].set_title("Steering Smoothness")
            axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig("eval_plot.png", dpi=150)
        print("Plot saved to eval_plot.png")
    plt.close()


def plot_trajectories(episodes_data, waypoints, save_dir, run_name="eval"):
    """
    Generate trajectory visualisations from recorded episode data.
    Fully headless — uses the Agg backend, no display or X server required.

    Saves:
        <save_dir>/overview.png        — all episodes overlaid, colour-coded by outcome
        <save_dir>/ep_NNN.png          — one plot per episode with overtake marker
        <save_dir>/best_epN.gif        — animated GIF of the highest-return episode
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(save_dir, exist_ok=True)
    wps = np.array(waypoints)[:, :2] if waypoints is not None else None

    def draw_track(ax):
        if wps is not None:
            ax.plot(wps[:, 0], wps[:, 1], "k-", lw=0.8, alpha=0.2, zorder=0)
            # Close the loop
            ax.plot([wps[-1, 0], wps[0, 0]], [wps[-1, 1], wps[0, 1]], "k-", lw=0.8, alpha=0.2, zorder=0)

    # ------------------------------------------------------------------ #
    # Overview: all ego trajectories overlaid, colour-coded by outcome
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_track(ax)
    for ep in episodes_data:
        frames = ep["frames"]
        ego_xs = [f["ego_x"] for f in frames]
        ego_ys = [f["ego_y"] for f in frames]
        color = "green" if ep["overtake"] else "red" if ep["collision"] else "steelblue"
        ax.plot(ego_xs, ego_ys, color=color, lw=0.8, alpha=0.45, zorder=1)

    n_ot = sum(1 for e in episodes_data if e["overtake"])
    n_cr = sum(1 for e in episodes_data if e["collision"])
    ax.legend(handles=[
        Line2D([0], [0], color="green",    lw=2, label=f"Overtake ({n_ot})"),
        Line2D([0], [0], color="red",      lw=2, label=f"Collision ({n_cr})"),
        Line2D([0], [0], color="steelblue", lw=2, label=f"Timeout ({len(episodes_data)-n_ot-n_cr})"),
    ], loc="best")
    ax.set_title(f"{run_name}  —  {n_ot}/{len(episodes_data)} overtakes", fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")
    overview_path = os.path.join(save_dir, "overview.png")
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview:  {overview_path}")

    # ------------------------------------------------------------------ #
    # Per-episode plots
    # ------------------------------------------------------------------ #
    for ep_idx, ep in enumerate(episodes_data):
        frames = ep["frames"]
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_track(ax)

        ego_xs = [f["ego_x"] for f in frames]
        ego_ys = [f["ego_y"] for f in frames]
        ax.plot(ego_xs, ego_ys, "b-", lw=1.5, label="Ego (RL)", zorder=2)
        ax.plot(ego_xs[0], ego_ys[0], "bs", ms=8, zorder=3)   # start
        ax.plot(ego_xs[-1], ego_ys[-1], "bx", ms=8, zorder=3) # end

        has_opp = frames[0]["opp_x"] is not None
        if has_opp:
            opp_xs = [f["opp_x"] for f in frames]
            opp_ys = [f["opp_y"] for f in frames]
            ax.plot(opp_xs, opp_ys, "r-", lw=1.5, label="Opponent (PP)", zorder=2)
            ax.plot(opp_xs[0], opp_ys[0], "rs", ms=8, zorder=3)

        if ep["overtake_step"] is not None:
            ot = ep["overtake_step"]
            ax.plot(ego_xs[ot], ego_ys[ot], "g*", ms=18, zorder=4,
                    label=f"Overtake (step {ot})")

        outcome = "OVERTAKE" if ep["overtake"] else "CRASH" if ep["collision"] else "timeout"
        ax.set_title(f"Episode {ep_idx+1}  |  Return: {ep['return']:.1f}  |  {outcome}")
        ax.legend(loc="best")
        ax.set_aspect("equal")
        ax.axis("off")
        ep_path = os.path.join(save_dir, f"ep_{ep_idx+1:03d}.png")
        fig.savefig(ep_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(episodes_data)} episode plots to {save_dir}/")

    # ------------------------------------------------------------------ #
    # Animated GIF for the best episode
    # ------------------------------------------------------------------ #
    best = max(episodes_data, key=lambda e: e["return"])
    best_idx = episodes_data.index(best)
    gif_path = os.path.join(save_dir, f"best_ep{best_idx+1}.gif")
    _animate_episode(best, wps, gif_path)


def _animate_episode(ep_data, wps, save_path):
    """
    Save an animated GIF of car positions for one episode.
    Uses PillowWriter — requires only Pillow, no display needed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = ep_data["frames"]
    # Subsample to ≤300 frames for a reasonable file size
    step = max(1, len(frames) // 300)
    sampled = frames[::step]

    fig, ax = plt.subplots(figsize=(7, 7))
    if wps is not None:
        ax.plot(wps[:, 0], wps[:, 1], "k-", lw=0.8, alpha=0.2)
        ax.plot([wps[-1, 0], wps[0, 0]], [wps[-1, 1], wps[0, 1]], "k-", lw=0.8, alpha=0.2)

    # Faint ghost trajectories for context
    ego_xs_all = [f["ego_x"] for f in frames]
    ego_ys_all = [f["ego_y"] for f in frames]
    ax.plot(ego_xs_all, ego_ys_all, "b-", lw=0.5, alpha=0.12)

    has_opp = frames[0]["opp_x"] is not None
    if has_opp:
        ax.plot([f["opp_x"] for f in frames], [f["opp_y"] for f in frames],
                "r-", lw=0.5, alpha=0.12)

    ego_dot,  = ax.plot([], [], "bo", ms=10, label="Ego (RL)")
    opp_dot,  = ax.plot([], [], "ro", ms=10, label="Opponent")
    title_obj = ax.set_title("")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.axis("off")

    overtake_step = ep_data.get("overtake_step")

    def update(i):
        f = sampled[i]
        ego_dot.set_data([f["ego_x"]], [f["ego_y"]])
        if has_opp and f["opp_x"] is not None:
            opp_dot.set_data([f["opp_x"]], [f["opp_y"]])
        real_step = i * step
        flag = "  ★ OVERTAKE!" if (overtake_step is not None and real_step >= overtake_step) else ""
        title_obj.set_text(f"Step {real_step}{flag}")
        return ego_dot, opp_dot, title_obj

    ani = FuncAnimation(fig, update, frames=len(sampled), interval=50, blit=False)
    try:
        ani.save(save_path, writer=PillowWriter(fps=20))
        print(f"  Saved animation: {save_path}")
    except Exception as e:
        print(f"  Animation save failed: {e}  (try: pip install Pillow)")
    plt.close(fig)


def export_onnx(model_path, config):
    """Export model to ONNX."""
    import torch
    from stable_baselines3 import PPO, SAC, TD3

    algo_type = config["algorithm"]["type"]
    AlgoClass = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo_type]
    model = AlgoClass.load(model_path, device="cpu")
    obs_dim = model.observation_space.shape[0]

    onnx_path = model_path.replace(".zip", "") + ".onnx"
    dummy = torch.randn(1, obs_dim)
    torch.onnx.export(model.policy, dummy, onnx_path, opset_version=11,
                        input_names=["observation"], output_names=["action"])
    print(f"Exported ONNX model to {onnx_path}")


def evaluate_bc_model(bc_model_path, config, args):
    """Evaluate a standalone behavioral cloning model."""
    import torch
    from f1tenth_rl.envs.wrapper import F1TenthWrapper
    from f1tenth_rl.agents.networks import RacingMLP

    render_mode = "human" if args.render else None
    env = F1TenthWrapper(config, render_mode=render_mode)

    # Load BC model
    checkpoint = torch.load(bc_model_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]

    # Use config from checkpoint if available and no override
    if "config" in checkpoint and not args.config:
        config = checkpoint["config"]
        print(f"  Using config from checkpoint")

    # Infer dimensions from the first layer
    first_key = [k for k in model_state.keys() if "weight" in k][0]
    obs_dim = model_state[first_key].shape[1]
    last_key = [k for k in model_state.keys() if "weight" in k][-1]
    act_dim = model_state[last_key].shape[0]

    # Rebuild model
    import torch.nn as nn
    hidden = config.get("network", {}).get("mlp", {}).get("hidden_sizes", [256, 256])
    model = nn.Sequential(
        RacingMLP(obs_dim, act_dim, hidden_sizes=hidden),
        nn.Tanh(),
    )
    model.load_state_dict(model_state)
    model.eval()

    print(f"\nEvaluating BC model: {bc_model_path}")
    print(f"  Obs dim:  {obs_dim}")
    print(f"  Act dim:  {act_dim}")
    print(f"  Episodes: {args.episodes}")
    print("-" * 50)

    metrics = defaultdict(list)

    for ep in range(args.episodes):
        obs, info = env.reset()
        episode_return = 0
        speeds = []
        steers = []
        done = False

        while not done:
            # 1. Normalize Observation
            obs_input = obs
            if obs_rms is not None:
                obs_input = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
                obs_input = np.clip(obs_input, -10.0, 10.0).astype(np.float32)

            # 2. Predict Action
            action, _ = model.predict(obs_input, deterministic=True)

            print("action obtained: ")

            # 3. Step Environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

            speeds.append(info.get("ego_speed", 0))
            steers.append(abs(info.get("physical_action",)))

            if args.render:
                env.render()
                time.sleep(0.005)

        metrics["return"].append(episode_return)
        metrics["length"].append(info.get("episode_steps", 0))
        metrics["speed"].append(np.mean(speeds) if speeds else 0)
        metrics["steer_smooth"].append(np.mean(steers) if steers else 0)
        progress = info.get("progress", 0)
        metrics["progress"].append(progress)
        collision = info.get("collision", False)
        metrics["collision"].append(float(collision))
        lap_time = info.get("lap_time", 0)
        if lap_time > 0:
            metrics["lap_time"].append(lap_time)

        status = "CRASH" if collision else f"progress={progress:.1%}"
        print(f"  Ep {ep+1:3d}: reward={episode_return:7.1f} | {status}")

    env.close()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate F1TENTH RL policies")
    parser.add_argument("--run", type=str, nargs="+", default=None,
                        help="Run directory/directories to evaluate")
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Model path(s) to evaluate (legacy)")
    parser.add_argument("--bc-model", type=str, default=None,
                        help="Path to behavioral cloning .pt model")
    parser.add_argument("--use-best", action="store_true",
                        help="Evaluate best_model instead of final_model")
    parser.add_argument("--config", type=str, default=None, help="Override config")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true",
                        help="Live rendering (requires a display; not available on headless VMs)")
    parser.add_argument("--record", action="store_true",
                        help="Record trajectory plots + animated GIF (headless, no display needed)")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Where to save recordings (default: <run_dir>/eval_recordings/)")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)
    args = parser.parse_args()

    # ---- BC model evaluation ----
    if args.bc_model:
        config_path = args.config or str(project_root / "configs" / "default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        metrics = evaluate_bc_model(args.bc_model, config, args)
        print_results(metrics, "BC Model")
        return

    sources = args.run or args.model
    if not sources:
        parser.error("Provide --run or --model")

    all_metrics = []
    names = []

    for src in sources:
        model_path, config_path, norm_path = find_model_and_config(src, args.use_best)

        # Load config
        if args.config:
            config_path = args.config
        if not os.path.exists(config_path):
            config_path = str(project_root / "configs" / "default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if args.export_onnx:
            export_onnx(model_path, config)
            continue

        metrics, episodes_data, waypoints = evaluate_model(model_path, config, norm_path, args)
        print_results(metrics, os.path.basename(src))
        all_metrics.append(metrics)
        names.append(os.path.basename(src))

        if args.record and episodes_data:
            rec_dir = args.record_dir or str(Path(src) / "eval_recordings")
            print(f"\nSaving recordings to: {rec_dir}")
            plot_trajectories(episodes_data, waypoints, rec_dir, run_name=os.path.basename(src))

    if (args.plot or args.save_plot) and all_metrics:
        plot_results(all_metrics, names, args.save_plot)


if __name__ == "__main__":
    main()
