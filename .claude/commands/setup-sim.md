Set up a drone racing simulator for development and training.

Available simulators to configure:
1. **gym-pybullet-drones** (recommended for RL training) - PyBullet-based, fast, Python-native
2. **lsy_drone_racing** (recommended for realistic sim) - CrazyFlow-based, high fidelity
3. **AirSim** (optional, heavyweight) - Unreal Engine, photorealistic

Steps:
1. Check what's already installed
2. Based on `$ARGUMENTS` (or default to "pybullet"):
   - "pybullet": `pip install gym-pybullet-drones`
   - "lsy": `pip install crazyflow` and clone utiasDSL/lsy_drone_racing
   - "airsim": Install AirSim Python package
3. Create a test script that:
   - Initializes the chosen simulator
   - Spawns a drone
   - Runs a simple hover test
   - Verifies the sim is working
4. Update the sim_interface.py to support the new simulator

Report setup status and any issues.
