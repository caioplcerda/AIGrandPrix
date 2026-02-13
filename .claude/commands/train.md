Set up and run reinforcement learning training for the drone racing agent.

Steps:
1. Check if the virtual environment exists, create one if not: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
2. Verify all dependencies are installed
3. Run the RL training script with the following approach:
   - Use stable-baselines3 PPO algorithm
   - Use the DroneRacingEnv from src/aigrandprix/simulation/gym_env.py
   - Load config from configs/default.yaml
   - Save checkpoints to models/checkpoints/
   - Log training metrics

If the user provides arguments like `$ARGUMENTS`, use them to override training parameters (e.g., total_timesteps, learning_rate).

After training completes, report:
- Total training time
- Final reward metrics
- Best checkpoint saved location
