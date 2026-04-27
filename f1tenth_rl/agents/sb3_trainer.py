"""
Stable Baselines 3 Trainer
===========================
Handles training with SB3 algorithms (PPO, SAC, TD3).

Run organization:
    runs/
    └── ppo_spielberg_2026-04-04_18-30-00/
        ├── config.yaml              # Full config snapshot
        ├── checkpoints/             # Periodic checkpoints
        │   ├── model_50000_steps.zip
        │   └── model_50000_steps_vecnormalize.pkl
        ├── best_model/              # Best eval model
        │   └── best_model.zip
        ├── final_model.zip          # Final model
        ├── final_vecnormalize.pkl   # Final normalization stats
        └── eval/                    # Eval logs
"""

import os
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import VecNormalize

from f1tenth_rl.envs.wrapper import make_vec_env, make_env
from f1tenth_rl.agents.networks import get_policy_kwargs
from f1tenth_rl.utils.callbacks import RacingMetricsCallback, WandbSafeCallback, CurriculumDRCallback, SelfPlayCallback


class SB3Trainer:
    """
    Training manager for Stable Baselines 3 algorithms.

    Creates an organized run directory with config, checkpoints,
    best model, final model, and optional WandB logging.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    """

    ALGORITHMS = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algo_type = config["algorithm"]["type"]
        self.total_timesteps = config["algorithm"]["total_timesteps"]
        self.seed = config["experiment"].get("seed", 42)
        self.device = config["experiment"].get("device", "auto")

        # ---- Build run directory name ----
        exp_name = config["experiment"].get("name", "")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if exp_name:
            self.run_name = f"{exp_name}_{timestamp}"
        else:
            map_name = Path(config["env"]["map_path"]).stem
            self.run_name = f"{self.algo_type}_{map_name}_{timestamp}"

        # ---- Create run directory structure ----
        runs_dir = config["experiment"].get("runs_dir", "runs")
        self.run_dir = os.path.join(runs_dir, self.run_name)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.best_model_dir = os.path.join(self.run_dir, "best_model")
        self.eval_dir = os.path.join(self.run_dir, "eval")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # Save config immediately
        config_path = os.path.join(self.run_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Build environments
        self.train_env = None
        self.eval_env = None
        self.model = None
        self.wandb_run = None

    def _apply_algo_overrides(self):
        """Merge {algo_type}_overrides into base config sections before env creation.

        Allows per-algorithm tuning of reward, action, and env settings in the YAML
        without losing the values that work for other algorithms.
        """
        for section in ("reward", "action", "env"):
            sec_cfg = self.config.get(section, {})
            overrides = sec_cfg.pop(f"{self.algo_type}_overrides", {})
            if overrides:
                print(f"  Applying {self.algo_type} overrides to [{section}]: {overrides}")
                sec_cfg.update(overrides)

    def setup(self):
        """Create environments and algorithm. Call before train()."""
        self._apply_algo_overrides()

        n_envs = self.config["env"].get("num_envs", 8)
        self.train_env = make_vec_env(
            self.config, n_envs=n_envs, seed=self.seed, normalize=True
        )
        self.eval_env = make_vec_env(
            self.config, n_envs=1, seed=self.seed + 1000, normalize=True
        )

        # ---- Algorithm hyperparameters ----
        algo_cfg = self.config["algorithm"].get(self.algo_type, {})
        policy_kwargs = get_policy_kwargs(self.config["network"], algo_type=self.algo_type)

        AlgoClass = self.ALGORITHMS.get(self.algo_type)
        if AlgoClass is None:
            raise ValueError(
                f"Unknown algorithm: {self.algo_type}. "
                f"Choose from: {list(self.ALGORITHMS.keys())}"
            )

        common_kwargs = {
            "policy": "MlpPolicy",
            "env": self.train_env,
            "seed": self.seed,
            "device": self.device,
            "verbose": 1,
            "tensorboard_log": self.tb_dir,
            "policy_kwargs": policy_kwargs,
        }

        if self.algo_type == "ppo":
            self.model = AlgoClass(
                **common_kwargs,
                learning_rate=algo_cfg.get("learning_rate", 3e-4),
                n_steps=algo_cfg.get("n_steps", 2048),
                batch_size=algo_cfg.get("batch_size", 128),
                n_epochs=algo_cfg.get("n_epochs", 10),
                gamma=algo_cfg.get("gamma", 0.99),
                gae_lambda=algo_cfg.get("gae_lambda", 0.95),
                clip_range=algo_cfg.get("clip_range", 0.2),
                ent_coef=algo_cfg.get("ent_coef", 0.01),
                vf_coef=algo_cfg.get("vf_coef", 0.5),
                max_grad_norm=algo_cfg.get("max_grad_norm", 0.5),
                use_sde=algo_cfg.get("use_sde", False),
            )
        elif self.algo_type == "sac":
            self.model = AlgoClass(
                **common_kwargs,
                learning_rate=algo_cfg.get("learning_rate", 3e-4),
                buffer_size=algo_cfg.get("buffer_size", 100_000),
                batch_size=algo_cfg.get("batch_size", 256),
                tau=algo_cfg.get("tau", 0.005),
                gamma=algo_cfg.get("gamma", 0.99),
                learning_starts=algo_cfg.get("learning_starts", 20000),
                train_freq=algo_cfg.get("train_freq", 1),
                gradient_steps=algo_cfg.get("gradient_steps", 1),
                ent_coef=algo_cfg.get("ent_coef", 0.1),
            )
        elif self.algo_type == "td3":
            self.model = AlgoClass(
                **common_kwargs,
                learning_rate=algo_cfg.get("learning_rate", 1e-3),
                buffer_size=algo_cfg.get("buffer_size", 300_000),
                batch_size=algo_cfg.get("batch_size", 256),
                tau=algo_cfg.get("tau", 0.005),
                gamma=algo_cfg.get("gamma", 0.99),
                learning_starts=algo_cfg.get("learning_starts", 10000),
                train_freq=algo_cfg.get("train_freq", 1),
                gradient_steps=algo_cfg.get("gradient_steps", 1),
                policy_delay=algo_cfg.get("policy_delay", 2),
            )

        print(f"\n{'='*60}")
        print(f"  F1TENTH RL Training")
        print(f"{'='*60}")
        print(f"  Run:          {self.run_name}")
        print(f"  Run dir:      {self.run_dir}")
        print(f"  Algorithm:    {self.algo_type.upper()}")
        print(f"  Map:          {self.config['env']['map_path']}")
        print(f"  Environments: {n_envs}")
        print(f"  Total steps:  {self.total_timesteps:,}")
        print(f"  Device:       {self.device}")
        print(f"  Seed:         {self.seed}")
        print(f"  Network:      {self.config['network']['type']}")
        print(f"  Obs space:    {self.train_env.observation_space.shape}")
        print(f"  Act space:    {self.train_env.action_space.shape}")
        print(f"  WandB:        {self.config['experiment'].get('wandb', False)}")
        print(f"{'='*60}\n")

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            wandb_cfg = self.config["experiment"]
            self.wandb_run = wandb.init(
                project=wandb_cfg.get("wandb_project", "f1tenth_rl"),
                entity=wandb_cfg.get("wandb_entity", None),
                name=self.run_name,
                config=self.config,
                dir=self.run_dir,
                sync_tensorboard=True,
                save_code=True,
                tags=[
                    self.algo_type,
                    Path(self.config["env"]["map_path"]).stem,
                    self.config["observation"]["type"],
                    self.config["reward"]["type"],
                ],
            )

            # Log additional summary info
            wandb.config.update({
                "obs_dim": self.train_env.observation_space.shape[0],
                "act_dim": self.train_env.action_space.shape[0],
                "num_envs": self.config["env"].get("num_envs", 8),
                "run_dir": self.run_dir,
            }, allow_val_change=True)

            print(f"  WandB run: {wandb.run.get_url()}")
            return True

        except ImportError:
            print("[WARNING] wandb not installed. Run: pip install wandb")
            return False

    def train(self):
        """Run the training loop (single-map or multi-map sequential)."""
        if self.model is None:
            self.setup()

        use_wandb = self.config["experiment"].get("wandb", False)
        if use_wandb:
            use_wandb = self._init_wandb()

        multi_map_cfg = self.config.get("multi_map", {})
        maps = multi_map_cfg.get("maps", None)

        if maps:
            self._train_multi_map(maps, multi_map_cfg, use_wandb)
        else:
            self._train_single(use_wandb)

        if use_wandb:
            self._log_wandb_artifacts()
            import wandb
            wandb.finish()

        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Run directory: {self.run_dir}")
        print(f"  Final model:   {self.run_dir}/final_model.zip")
        print(f"  Best model:    {self.best_model_dir}/best_model.zip")
        print(f"{'='*60}\n")

    def _train_single(self, use_wandb: bool):
        """Single-map training (existing behaviour)."""
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=CallbackList(self._build_callbacks(use_wandb)),
            progress_bar=True,
        )
        self._save_final()

    def _train_multi_map(self, maps, cfg, use_wandb: bool):
        """Sequential multi-map training. Step counter is global across all maps."""
        n_maps = len(maps)
        steps_per_map = cfg.get("timesteps_per_map", self.total_timesteps // n_maps)

        steps_done = 0
        for i, map_path in enumerate(maps):
            print(f"\n{'='*60}")
            print(f"  Map phase {i+1}/{n_maps}: {map_path}")
            print(f"  Steps this phase: {steps_per_map:,}  |  "
                  f"Global target: {steps_done + steps_per_map:,}")
            print(f"{'='*60}")

            if i > 0 or self.config["env"]["map_path"] != map_path:
                self._swap_map(map_path)

            self.model.learn(
                total_timesteps=steps_done + steps_per_map,
                callback=CallbackList(self._build_callbacks(use_wandb)),
                reset_num_timesteps=(i == 0),
                progress_bar=True,
            )
            steps_done = self.model.num_timesteps

            # Boundary checkpoint — allows resuming from any map transition
            stem = Path(map_path).stem
            ckpt = os.path.join(self.checkpoint_dir, f"map{i+1:02d}_{stem}_final")
            self.model.save(ckpt)
            if isinstance(self.train_env, VecNormalize):
                self.train_env.save(ckpt + "_vecnormalize.pkl")
            print(f"  Saved boundary checkpoint: {ckpt}.zip")

        self._save_final()

    def _swap_map(self, map_path: str):
        """Replace train/eval envs for a new map, preserving VecNormalize reward stats.

        Uses a temp-save/reload cycle to handle any n_envs mismatch between the
        currently loaded model and the new environment (e.g. after --resume loads
        the model with n_envs=1 for the eval env).
        """
        import copy

        old_ret_rms = None
        if isinstance(self.train_env, VecNormalize):
            old_ret_rms = copy.deepcopy(self.train_env.ret_rms)

        # Save weights before closing envs so we can reload with the new env
        tmp_path = os.path.join(self.checkpoint_dir, "_swap_tmp")
        self.model.save(tmp_path)

        self.train_env.close()
        if self.eval_env is not None:
            self.eval_env.close()

        self.config["env"]["map_path"] = map_path

        n_envs = self.config["env"].get("num_envs", 8)
        self.train_env = make_vec_env(self.config, n_envs=n_envs,
                                      seed=self.seed, normalize=True)
        self.eval_env  = make_vec_env(self.config, n_envs=1,
                                      seed=self.seed + 1000, normalize=True)

        # Restore reward running stats — keeps normalisation scale stable across maps
        if old_ret_rms is not None and isinstance(self.train_env, VecNormalize):
            self.train_env.ret_rms = old_ret_rms

        # Reload with the new env so model.n_envs matches train_env.num_envs
        AlgoClass = self.ALGORITHMS[self.algo_type]
        self.model = AlgoClass.load(
            tmp_path, env=self.train_env, device=self.device,
            custom_objects=self._replay_buffer_overrides(),
        )

        tmp_zip = tmp_path + ".zip"
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

        print(f"  Swapped to map: {map_path}")

    def _build_callbacks(self, use_wandb: bool):
        """Create training callbacks."""
        callbacks = []
        cb_cfg = self.config.get("callbacks", {})
        n_envs = self.config["env"].get("num_envs", 8)

        # ---- Checkpoint callback ----
        checkpoint_freq = cb_cfg.get("checkpoint_freq", 50000)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(checkpoint_freq // n_envs, 1),
                save_path=self.checkpoint_dir,
                name_prefix="model",
                save_vecnormalize=True,
            )
        )

        # ---- Evaluation callback (saves best model) ----
        eval_cfg = self.config.get("evaluation", {})
        eval_freq = eval_cfg.get("eval_freq", 50000)
        callbacks.append(
            EvalCallback(
                self.eval_env,
                best_model_save_path=self.best_model_dir,
                log_path=self.eval_dir,
                eval_freq=max(eval_freq // n_envs, 1),
                n_eval_episodes=eval_cfg.get("n_eval_episodes", 10),
                deterministic=eval_cfg.get("deterministic", True),
            )
        )

        # ---- Racing metrics callback ----
        callbacks.append(RacingMetricsCallback(use_wandb=use_wandb))

        # ---- Curriculum domain randomization callback ----
        dr_cfg = self.config.get("domain_randomization", {})
        dr_mode = dr_cfg.get("mode", "fixed" if dr_cfg.get("enabled", False) else "off")
        if dr_mode == "curriculum":
            callbacks.append(CurriculumDRCallback(
                total_timesteps=self.total_timesteps,
                use_wandb=use_wandb,
                warmup=dr_cfg.get("curriculum_warmup", 0.2),
                full=dr_cfg.get("curriculum_full", 0.6),
            ))

        # ---- Self-play callback (RL vs RL) ----
        ma_cfg = self.config.get("multi_agent", {})
        if ma_cfg.get("opponent") == "self_play":
            callbacks.append(SelfPlayCallback(
                update_freq=ma_cfg.get("self_play_update_freq", 50000),
                use_wandb=use_wandb,
            ))

        # ---- WandB callback ----
        if use_wandb:
            callbacks.append(WandbSafeCallback())

        return callbacks

    def _save_final(self):
        """Save the final model, normalization stats, and config."""
        final_model_path = os.path.join(self.run_dir, "final_model")
        self.model.save(final_model_path)

        if isinstance(self.train_env, VecNormalize) and self.train_env.norm_obs:
            norm_path = os.path.join(self.run_dir, "final_vecnormalize.pkl")
            self.train_env.save(norm_path)

    def _log_wandb_artifacts(self):
        """Upload models to WandB as artifacts."""
        try:
            import wandb

            # Log final model
            artifact = wandb.Artifact(
                name=f"model-{self.run_name}",
                type="model",
                description=f"Final {self.algo_type.upper()} model",
            )
            artifact.add_file(os.path.join(self.run_dir, "final_model.zip"))
            norm_path = os.path.join(self.run_dir, "final_vecnormalize.pkl")
            if os.path.exists(norm_path):
                artifact.add_file(norm_path)
            artifact.add_file(os.path.join(self.run_dir, "config.yaml"))
            wandb.log_artifact(artifact)

            # Log best model if it exists
            best_path = os.path.join(self.best_model_dir, "best_model.zip")
            if os.path.exists(best_path):
                best_artifact = wandb.Artifact(
                    name=f"best-model-{self.run_name}",
                    type="model",
                    description=f"Best eval {self.algo_type.upper()} model",
                )
                best_artifact.add_file(best_path)
                wandb.log_artifact(best_artifact)

        except Exception as e:
            print(f"  [WandB] Failed to log artifacts: {e}")

    def save(self, path: str):
        """Save model and normalization statistics to a custom path."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        if isinstance(self.train_env, VecNormalize):
            self.train_env.save(path + "_vecnormalize.pkl")
        with open(path + "_config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    @staticmethod
    def _find_latest_checkpoint(checkpoint_dir: Path):
        """Return (model_stem, norm_path) for the highest-step checkpoint, or (None, None)."""
        import re
        best_step = -1
        best_stem = None
        for f in checkpoint_dir.glob("model_*_steps.zip"):
            m = re.search(r"model_(\d+)_steps\.zip", f.name)
            if m:
                step = int(m.group(1))
                if step > best_step:
                    best_step = step
                    best_stem = str(f).replace(".zip", "")
        if best_stem is None:
            return None, None
        norm = best_stem.replace("model_", "model_vecnormalize_") + ".pkl"
        norm_path = norm if os.path.exists(norm) else None
        return best_stem, norm_path

    def _replay_buffer_overrides(self) -> dict:
        """Return custom_objects dict to override saved buffer_size on SAC/TD3 load.

        SB3 restores the original buffer_size from the checkpoint by default.
        Passing custom_objects forces the new value from config instead.
        """
        if self.algo_type not in ("sac", "td3"):
            return {}
        algo_cfg = self.config["algorithm"].get(self.algo_type, {})
        buf = algo_cfg.get("buffer_size", 100_000)
        return {"buffer_size": buf}

    def load(self, path: str, env=None):
        """Load a trained model from a run directory or model path.

        When given a run directory, prefers final_model; falls back to the
        highest-step checkpoint if final_model does not exist.
        """
        AlgoClass = self.ALGORITHMS[self.algo_type]

        run_dir = Path(path)
        if run_dir.is_dir():
            final = run_dir / "final_model.zip"
            if final.exists():
                model_path = str(run_dir / "final_model")
                norm_path = str(run_dir / "final_vecnormalize.pkl")
            else:
                ckpt_dir = run_dir / "checkpoints"
                model_path, norm_path = self._find_latest_checkpoint(ckpt_dir)
                if model_path is None:
                    raise FileNotFoundError(
                        f"No final_model.zip and no step checkpoints found in {run_dir}"
                    )
                print(f"  No final_model found — resuming from latest checkpoint: {model_path}")
                if norm_path is None:
                    norm_path = str(run_dir / "final_vecnormalize.pkl")
        else:
            model_path = path
            norm_path = path + "_vecnormalize.pkl"

        if env is None:
            env = make_vec_env(self.config, n_envs=1, seed=self.seed, normalize=True)

        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False

        self.model = AlgoClass.load(
            model_path, env=env, device=self.device,
            custom_objects=self._replay_buffer_overrides(),
        )
        self.eval_env = env
        print(f"  Loaded model from {model_path}")

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Get action from trained model."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def close(self):
        """Clean up environments."""
        if self.train_env is not None:
            self.train_env.close()
        if self.eval_env is not None:
            self.eval_env.close()
