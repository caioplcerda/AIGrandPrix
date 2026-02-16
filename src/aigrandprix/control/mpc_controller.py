"""Linear MPC controller for drone racing.

Uses a linearized model of the simplified gym_env dynamics to solve
a quadratic program at each timestep for optimal trajectory tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from aigrandprix.control.drone_controller import ControlCommand, DroneState


@dataclass
class MPCConfig:
    """MPC controller configuration."""

    horizon: int = 10
    dt: float = 0.02
    Q_pos: list[float] = field(default_factory=lambda: [10.0, 10.0, 20.0])
    Q_vel: list[float] = field(default_factory=lambda: [2.0, 2.0, 5.0])
    R: list[float] = field(default_factory=lambda: [0.1, 0.01, 0.01])
    Q_terminal_scale: float = 2.0
    max_solve_time_ms: float = 5.0

    @classmethod
    def from_config(cls, config: dict) -> MPCConfig:
        return cls(
            horizon=config.get("horizon", 10),
            dt=config.get("dt", 0.02),
            Q_pos=config.get("Q_pos", [10.0, 10.0, 20.0]),
            Q_vel=config.get("Q_vel", [2.0, 2.0, 5.0]),
            R=config.get("R", [0.1, 0.01, 0.01]),
            Q_terminal_scale=config.get("Q_terminal_scale", 2.0),
            max_solve_time_ms=config.get("max_solve_time_ms", 5.0),
        )


class SimplifiedLinearModel:
    """Discrete LTI model matching gym_env simplified physics.

    State x = [pos(3), vel(3)] (6D)
    Control u = [thrust, roll_rate, pitch_rate] (3D)

    gym_env simplified physics:
        accel = [pitch_rate * 0.5, -roll_rate * 0.5, (thrust - 0.5) * 20.0 - 9.81]
        vel += accel * dt
        vel *= drag  (drag = 0.99)
        pos += vel * dt

    Rewritten as x_{k+1} = A*x_k + B*u_k + d
    """

    def __init__(self, dt: float = 0.02, drag: float = 0.99) -> None:
        self.dt = dt
        self.drag = drag
        self.nx = 6  # state dim
        self.nu = 3  # control dim

        # Build A matrix (6x6)
        self.A = np.zeros((6, 6))
        # pos_{k+1} = pos_k + drag * vel_k * dt  (vel after drag applied at step k)
        # Actually: vel_new = (vel + accel*dt) * drag; pos_new = pos + vel_new * dt
        # So pos depends on pos (I) and vel (drag*dt*I)
        self.A[:3, :3] = np.eye(3)
        self.A[:3, 3:6] = drag * dt * np.eye(3)
        # vel_{k+1} = drag * vel_k + drag * B_u * u_k * dt + drag * c * dt
        self.A[3:6, 3:6] = drag * np.eye(3)

        # B_u maps control to acceleration:
        # accel_x = pitch_rate * 0.5  → B_u[0,2] = 0.5
        # accel_y = -roll_rate * 0.5  → B_u[1,1] = -0.5
        # accel_z = thrust * 20.0     → B_u[2,0] = 20.0
        # (the -0.5*20 and -9.81 go into the affine term d)
        self._B_u = np.array([
            [0.0, 0.0, 0.5],   # accel_x from [thrust, roll_rate, pitch_rate]
            [0.0, -0.5, 0.0],  # accel_y
            [20.0, 0.0, 0.0],  # accel_z
        ])

        # B matrix (6x3): effect of control on state
        # vel part: drag * B_u * dt
        # pos part: drag * B_u * dt * dt  (through the vel->pos coupling)
        self.B = np.zeros((6, 3))
        self.B[3:6, :] = drag * self._B_u * dt
        self.B[:3, :] = drag * self._B_u * dt * dt  # second order pos effect

        # Affine term d (constant offset from gravity and thrust offset)
        # c = [0, 0, -0.5*20 - 9.81] = [0, 0, -19.81]
        c = np.array([0.0, 0.0, -0.5 * 20.0 - 9.81])
        self.d = np.zeros(6)
        self.d[3:6] = drag * c * dt
        self.d[:3] = drag * c * dt * dt

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Single-step state prediction."""
        return self.A @ x + self.B @ u + self.d

    def build_prediction_matrices(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build batch prediction matrices: X = S*x0 + T*U + W.

        X is (N*nx,) stacked states [x_1, x_2, ..., x_N]
        U is (N*nu,) stacked controls [u_0, u_1, ..., u_{N-1}]

        S: (N*nx, nx) — effect of initial state
        T: (N*nx, N*nu) — effect of control sequence
        W: (N*nx,) — affine offset (gravity etc.)
        """
        nx, nu = self.nx, self.nu

        S = np.zeros((N * nx, nx))
        T = np.zeros((N * nx, N * nu))
        W = np.zeros(N * nx)

        # A^k powers
        A_pow = np.eye(nx)
        A_powers = [A_pow.copy()]
        for _ in range(N):
            A_pow = A_pow @ self.A
            A_powers.append(A_pow.copy())

        for i in range(N):
            # S: row block i = A^(i+1)
            S[i * nx:(i + 1) * nx, :] = A_powers[i + 1]

            # T: effect of u_j on x_{i+1} = A^(i-j) * B for j <= i
            for j in range(i + 1):
                T[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = A_powers[i - j] @ self.B

            # W: sum of A^k * d for k=0..i
            w_i = np.zeros(nx)
            for k in range(i + 1):
                w_i += A_powers[k] @ self.d
            W[i * nx:(i + 1) * nx] = w_i

        return S, T, W


class MPCController:
    """Linear MPC controller using QP for trajectory tracking."""

    def __init__(
        self,
        config: MPCConfig | None = None,
        max_speed: float = 15.0,
        max_rate: float = 8.0,
        max_thrust: float = 1.0,
    ) -> None:
        self.config = config or MPCConfig()
        self.max_speed = max_speed
        self.max_rate = max_rate
        self.max_thrust = max_thrust

        self.model = SimplifiedLinearModel(dt=self.config.dt)
        N = self.config.horizon
        nx = self.model.nx
        nu = self.model.nu

        # Build weight matrices
        q_diag = np.array(self.config.Q_pos + self.config.Q_vel)
        q_terminal = q_diag * self.config.Q_terminal_scale
        r_diag = np.array(self.config.R)

        # Q_full: block diagonal for N steps, with terminal weight on last step
        Q_blocks = []
        for i in range(N):
            if i == N - 1:
                Q_blocks.append(np.diag(q_terminal))
            else:
                Q_blocks.append(np.diag(q_diag))
        self._Q = np.block([
            [Q_blocks[i] if i == j else np.zeros((nx, nx)) for j in range(N)]
            for i in range(N)
        ])

        # R_full: block diagonal for N steps
        self._R = np.kron(np.eye(N), np.diag(r_diag))

        # Precompute prediction matrices
        self._S, self._T, self._W = self.model.build_prediction_matrices(N)

        # Precompute H = 2*(T^T Q T + R) and its inverse
        self._H = 2.0 * (self._T.T @ self._Q @ self._T + self._R)
        # Regularize for numerical stability
        self._H += np.eye(N * nu) * 1e-8
        try:
            self._H_inv = np.linalg.inv(self._H)
        except np.linalg.LinAlgError:
            self._H_inv = np.linalg.pinv(self._H)

        # Control bounds
        self._u_min = np.tile(np.array([0.0, -max_rate, -max_rate]), N)
        self._u_max = np.tile(np.array([max_thrust, max_rate, max_rate]), N)

        # Warm start
        self._prev_solution: np.ndarray | None = None

    def compute_control(
        self,
        state: DroneState,
        reference_trajectory: list,
    ) -> ControlCommand:
        """Compute MPC control given current state and reference trajectory points."""
        t_start = time.perf_counter()
        N = self.config.horizon
        nx = self.model.nx
        nu = self.model.nu

        # Build current state vector
        x0 = np.concatenate([state.position, state.velocity])

        # Build reference
        x_ref, u_ref = self._trajectory_to_references(reference_trajectory)

        # X_ref stacked
        X_ref = x_ref.flatten()
        U_ref = u_ref.flatten()

        # Error from free response: e = S*x0 + W - X_ref
        e = self._S @ x0 + self._W - X_ref

        # f = 2*(T^T Q e - R U_ref)
        f = 2.0 * (self._T.T @ self._Q @ e - self._R @ U_ref)

        # Check time budget
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > self.config.max_solve_time_ms:
            return self._fallback_command(reference_trajectory, state)

        # Unconstrained solve: U = -H_inv * f
        try:
            U = -self._H_inv @ f
        except Exception:
            return self._fallback_command(reference_trajectory, state)

        # Clip to bounds
        U = np.clip(U, self._u_min, self._u_max)

        # Warm start: shift for next call
        self._prev_solution = np.zeros_like(U)
        self._prev_solution[:-nu] = U[nu:]
        self._prev_solution[-nu:] = U[-nu:]

        # Extract first control
        u0 = U[:nu]

        return ControlCommand(
            thrust=float(np.clip(u0[0], 0.0, self.max_thrust)),
            roll_rate=float(np.clip(u0[1], -self.max_rate, self.max_rate)),
            pitch_rate=float(np.clip(u0[2], -self.max_rate, self.max_rate)),
            yaw_rate=0.0,
        )

    def _trajectory_to_references(
        self,
        trajectory_points: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert trajectory points to state and control references.

        Returns:
            x_ref: (N, 6) state references [pos, vel]
            u_ref: (N, 3) control feedforward [thrust, roll_rate, pitch_rate]
        """
        N = self.config.horizon
        x_ref = np.zeros((N, 6))
        u_ref = np.zeros((N, 3))

        for i in range(N):
            if i < len(trajectory_points):
                pt = trajectory_points[i]
                x_ref[i, :3] = pt.position
                x_ref[i, 3:6] = pt.velocity
                # Feedforward: convert desired acceleration to control via simplified inverse
                accel = pt.acceleration if pt.acceleration is not None else np.zeros(3)
                u_ref[i, 0] = (accel[2] + 9.81) / 20.0 + 0.5  # thrust
                u_ref[i, 1] = -accel[1] * 2.0  # roll_rate
                u_ref[i, 2] = accel[0] * 2.0    # pitch_rate
            else:
                # Pad with last available point
                x_ref[i] = x_ref[min(i, len(trajectory_points)) - 1]
                u_ref[i] = u_ref[min(i, len(trajectory_points)) - 1]

        return x_ref, u_ref

    def _fallback_command(self, reference_trajectory: list, state: DroneState) -> ControlCommand:
        """Return feedforward command from first trajectory point on solve failure."""
        if reference_trajectory:
            pt = reference_trajectory[0]
            accel = pt.acceleration if pt.acceleration is not None else np.zeros(3)
            thrust = float(np.clip((accel[2] + 9.81) / 20.0 + 0.5, 0.0, self.max_thrust))
            pitch_rate = float(np.clip(accel[0] * 2.0, -self.max_rate, self.max_rate))
            roll_rate = float(np.clip(-accel[1] * 2.0, -self.max_rate, self.max_rate))
            return ControlCommand(thrust=thrust, roll_rate=roll_rate, pitch_rate=pitch_rate, yaw_rate=0.0)
        # Hover
        return ControlCommand(thrust=0.99, roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0)

    def reset(self) -> None:
        """Clear warm-start state."""
        self._prev_solution = None
