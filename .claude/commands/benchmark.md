Run a benchmark evaluation of the current racing agent.

Steps:
1. Load the latest model checkpoint from models/checkpoints/
2. Run 10 episodes in the DroneRacingEnv
3. Collect metrics:
   - Average lap time
   - Gate completion rate (gates passed / total gates)
   - Average speed through course
   - Crash rate
   - Best lap time
   - Worst lap time
4. Compare against previous benchmark results if they exist in configs/benchmarks.json
5. Save results to configs/benchmarks.json with timestamp

If `$ARGUMENTS` is provided, use it as the model path instead of the latest checkpoint.

Display results in a formatted table.
