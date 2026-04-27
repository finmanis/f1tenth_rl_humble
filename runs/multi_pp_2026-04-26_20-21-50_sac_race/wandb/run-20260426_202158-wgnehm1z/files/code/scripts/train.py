#!/usr/bin/env python3
"""
F1TENTH RL Training Script
============================

Usage:
    # Train with default config (uses WandB if installed)
    python scripts/train.py

    # Name your run
    python scripts/train.py --name my_experiment

    # Quick test
    python scripts/train.py --num-envs 1 --total-steps 10000

    # Full training
    python scripts/train.py --algo ppo --total-steps 1000000 --num-envs 8

    # Disable WandB
    python scripts/train.py --no-wandb

    # SAC with domain randomization
    python scripts/train.py --algo sac --domain-randomization --name sac_dr

    # BC pretraining + PPO fine-tuning
    python scripts/train.py --bc-pretrain demos/expert_demos.npz

    # Multi-map sequential training (step counter is global across maps)
    python scripts/train.py --config configs/custom.yaml \\
        --maps maps/spielberg/Spielberg maps/levine_slam/levine_slam maps/levine_blocked/levine_blocked \\
        --steps-per-map 700000
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL policies for F1TENTH autonomous racing",
    )

    # Config
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")

    # Run identity
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (timestamp added automatically)")

    # Algorithm
    parser.add_argument("--algo", type=str, default=None,
                        choices=["ppo", "sac", "td3", "custom_ppo"])
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    # Environment
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)

    # Observation
    parser.add_argument("--obs-type", type=str, default=None,
                        choices=["lidar_only", "lidar_state", "lidar_waypoint", "waypoint_only"])
    parser.add_argument("--lidar-beams", type=int, default=None)
    parser.add_argument("--lidar-sensor", type=str, default=None,
                        choices=["hokuyo", "sick", "custom"],
                        help="LiDAR preset: hokuyo (1080 beams) or sick (811 beams)")
    parser.add_argument("--lidar-raw-beams", type=int, default=None,
                        help="Custom raw beam count (use with --lidar-sensor custom)")

    # Reward & network
    parser.add_argument("--reward-type", type=str, default=None,
                        choices=["progress", "cth", "speed"])
    parser.add_argument("--network", type=str, default=None,
                        choices=["mlp", "cnn1d"])

    # Features
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--dr-mode", type=str, default=None,
                        choices=["off", "fixed", "curriculum"],
                        help="Domain randomization mode")
    parser.add_argument("--bc-pretrain", type=str, default=None,
                        help="Path to expert demos for BC pretraining")
    parser.add_argument("--bc-only", action="store_true",
                        help="Train behavioral cloning only (no RL fine-tuning)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to run directory or checkpoint to resume")
    parser.add_argument("--maps", type=str, nargs="+", default=None,
                        help="Ordered map paths for sequential multi-map training")
    parser.add_argument("--steps-per-map", type=int, default=None,
                        help="Timesteps on each map (default: total_timesteps / n_maps)")

    # Logging
    parser.add_argument("--wandb", action="store_true", default=None,
                        help="Enable WandB logging (default: from config)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--wandb-project", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def apply_overrides(config, args):
    """Apply command-line overrides to config."""
    if args.name:
        config["experiment"]["name"] = args.name
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.device:
        config["experiment"]["device"] = args.device
    if args.algo:
        config["algorithm"]["type"] = args.algo
    if args.total_steps:
        config["algorithm"]["total_timesteps"] = args.total_steps
    if args.lr:
        algo = config["algorithm"]["type"]
        if algo in config["algorithm"]:
            config["algorithm"][algo]["learning_rate"] = args.lr
    if args.map:
        config["env"]["map_path"] = args.map
    if args.num_envs:
        config["env"]["num_envs"] = args.num_envs
    if args.num_agents:
        config["env"]["num_agents"] = args.num_agents
    if args.obs_type:
        config["observation"]["type"] = args.obs_type
    if args.lidar_beams:
        config["observation"]["lidar_beams"] = args.lidar_beams

    # LiDAR sensor presets
    LIDAR_PRESETS = {
        "hokuyo": {"raw_beams": 1080, "fov_deg": 270.0, "range_max": 30.0},
        "sick":   {"raw_beams": 811,  "fov_deg": 270.0, "range_max": 25.0},
    }
    if args.lidar_sensor:
        if "lidar" not in config:
            config["lidar"] = {}
        if args.lidar_sensor in LIDAR_PRESETS:
            config["lidar"].update(LIDAR_PRESETS[args.lidar_sensor])
        elif args.lidar_sensor == "custom" and args.lidar_raw_beams:
            config["lidar"]["raw_beams"] = args.lidar_raw_beams
    if args.lidar_raw_beams and not args.lidar_sensor:
        if "lidar" not in config:
            config["lidar"] = {}
        config["lidar"]["raw_beams"] = args.lidar_raw_beams
    if args.reward_type:
        config["reward"]["type"] = args.reward_type
    if args.network:
        config["network"]["type"] = args.network
    if args.domain_randomization:
        config["domain_randomization"]["enabled"] = True
        if not config["domain_randomization"].get("mode"):
            config["domain_randomization"]["mode"] = "fixed"
    if args.dr_mode:
        config["domain_randomization"]["mode"] = args.dr_mode
        if args.dr_mode != "off":
            config["domain_randomization"]["enabled"] = True

    # Multi-map sequential training
    if args.maps:
        if "multi_map" not in config:
            config["multi_map"] = {}
        config["multi_map"]["maps"] = args.maps
    if args.steps_per_map:
        if "multi_map" not in config:
            config["multi_map"] = {}
        config["multi_map"]["timesteps_per_map"] = args.steps_per_map

    # WandB: CLI flags override config
    if args.no_wandb:
        config["experiment"]["wandb"] = False
    elif args.wandb:
        config["experiment"]["wandb"] = True
    if args.wandb_project:
        config["experiment"]["wandb_project"] = args.wandb_project

    return config


def main():
    args = parse_args()

    # ---- Load config ----
    config_path = Path(project_root) / args.config
    if not config_path.exists():
        config_path = Path(project_root) / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = apply_overrides(config, args)
    algo_type = config["algorithm"]["type"]

    # ---- BC Pretraining (optional) ----
    bc_path = None
    if args.bc_pretrain:
        print("=" * 60)
        if args.bc_only:
            print("  Behavioral Cloning Training (standalone)")
        else:
            print("  Step 1: Behavioral Cloning Pretraining")
        print("=" * 60)
        from f1tenth_rl.agents.imitation import ImitationTrainer

        il_trainer = ImitationTrainer(config)
        il_trainer.load_demonstrations(args.bc_pretrain)
        il_trainer.train_bc()

        bc_path = os.path.join("runs", "bc_pretrained")
        os.makedirs(bc_path, exist_ok=True)
        il_trainer.save(os.path.join(bc_path, "bc_model"))

        if args.bc_only:
            print()
            print("=" * 60)
            print("  BC Training Complete!")
            print(f"  Model saved: {bc_path}/bc_model.pt")
            print()
            print("  Evaluate:")
            print(f"    python scripts/evaluate.py --bc-model {bc_path}/bc_model.pt --episodes 20 --render")
            print()
            print("  To fine-tune with RL:")
            print(f"    python scripts/train.py --bc-pretrain {args.bc_pretrain}")
            print("=" * 60)
            return
        print()

    # ---- RL Training ----
    if algo_type == "custom_ppo":
        from f1tenth_rl.agents.custom_ppo import CustomPPO
        ppo = CustomPPO(config)
        ppo.train()
    else:
        from f1tenth_rl.agents.sb3_trainer import SB3Trainer

        trainer = SB3Trainer(config)
        trainer.setup()

        # Initialize from BC if available
        if bc_path:
            from f1tenth_rl.agents.imitation import ImitationTrainer
            il = ImitationTrainer(config)
            il.init_sb3_from_bc(trainer.model, os.path.join(bc_path, "bc_model"))

        if args.resume:
            trainer.load(args.resume)

        trainer.train()
        trainer.close()


if __name__ == "__main__":
    main()
