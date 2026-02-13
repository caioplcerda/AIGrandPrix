Run a visual test flight of the racing agent in the simulator.

Steps:
1. Ensure dependencies are installed
2. Load the racing agent (latest checkpoint or specified model)
3. Run a single episode with render_mode="human" if display available, otherwise "rgb_array"
4. Save the flight trajectory data to a temporary file
5. Generate a matplotlib visualization showing:
   - 3D flight path
   - Gate positions
   - Speed profile over time
   - Gate passage events
6. Save the plot to outputs/flight_viz.png

If `$ARGUMENTS` is provided, use it as: number_of_gates difficulty_level

Report flight results: time, gates passed, average speed, any crashes.
