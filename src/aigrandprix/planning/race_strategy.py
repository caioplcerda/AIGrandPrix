"""Racing line optimization for drone racing.

Shifts gate crossing points off-center to reduce path curvature and total
distance, producing faster trajectories through gate sequences.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from aigrandprix.planning.path_planner import Gate


class RacingLineOptimizer:
    """Optimise gate crossing positions to shorten and smooth the racing line."""

    def __init__(
        self,
        gate_margin: float = 0.15,
        curvature_weight: float = 2.0,
    ) -> None:
        self.gate_margin = gate_margin
        self.curvature_weight = curvature_weight

    def gate_frame(self, gate: Gate) -> tuple[np.ndarray, np.ndarray]:
        """Compute orthonormal gate-plane vectors (right, up).

        right: horizontal vector in the gate plane
        up: vertical vector in the gate plane
        """
        normal = gate.normal / np.linalg.norm(gate.normal)
        world_up = np.array([0.0, 0.0, 1.0])

        # If normal is nearly vertical, use a different reference
        if abs(np.dot(normal, world_up)) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0])

        right = np.cross(normal, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, normal)
        up = up / np.linalg.norm(up)

        return right, up

    def optimize_gates(
        self,
        gates: list[Gate],
        start_pos: np.ndarray,
    ) -> list[Gate]:
        """Optimise gate crossing positions to minimise path length + curvature.

        Returns new Gate objects with shifted positions (normals unchanged).
        """
        n_gates = len(gates)
        if n_gates == 0:
            return []

        # Pre-compute gate frames
        frames = [self.gate_frame(g) for g in gates]

        # Bounds: offset in gate-local frame
        bounds = []
        for g in gates:
            u_max = g.width / 2.0 - self.gate_margin
            v_max = g.height / 2.0 - self.gate_margin
            bounds.append((-u_max, u_max))
            bounds.append((-v_max, v_max))

        def _positions_from_offsets(offsets: np.ndarray) -> list[np.ndarray]:
            positions = []
            for i in range(n_gates):
                u, v = offsets[2 * i], offsets[2 * i + 1]
                right, up = frames[i]
                pos = gates[i].position + u * right + v * up
                positions.append(pos)
            return positions

        def cost(offsets: np.ndarray) -> float:
            positions = _positions_from_offsets(offsets)
            # Build full path: start -> gate positions
            all_pts = [start_pos] + positions

            # Path length
            path_length = 0.0
            for i in range(len(all_pts) - 1):
                path_length += float(np.linalg.norm(all_pts[i + 1] - all_pts[i]))

            # Curvature penalty: angle at each interior waypoint
            curvature_cost = 0.0
            for i in range(1, len(all_pts) - 1):
                v1 = all_pts[i] - all_pts[i - 1]
                v2 = all_pts[i + 1] - all_pts[i]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-9 and n2 > 1e-9:
                    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    curvature_cost += angle ** 2

            return path_length + self.curvature_weight * curvature_cost

        x0 = np.zeros(2 * n_gates)

        result = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )

        optimized_positions = _positions_from_offsets(result.x)
        optimized_gates = []
        for i, gate in enumerate(gates):
            optimized_gates.append(replace(
                gate,
                position=optimized_positions[i],
            ))

        return optimized_gates
