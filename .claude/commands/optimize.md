Analyze and optimize the current racing agent's performance.

Steps:
1. Profile the agent's compute_action function for latency
2. Identify the slowest components (perception, planning, control)
3. Analyze the current trajectory planner's efficiency:
   - Is the path close to time-optimal?
   - Are there unnecessary speed reductions?
   - Can gate approach angles be improved?
4. Review controller gains and suggest tuning improvements
5. Check if the perception pipeline is a bottleneck
6. Suggest specific code changes to improve lap time

Focus areas based on `$ARGUMENTS`:
- "perception" - optimize gate detection speed/accuracy
- "control" - tune PID/MPC gains
- "planning" - improve trajectory generation
- "all" or empty - analyze everything

Provide concrete, actionable suggestions with code changes.
