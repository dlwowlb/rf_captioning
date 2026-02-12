"""Wall model for through-wall RF propagation simulation.

Models electromagnetic wave penetration through walls with:
- Fresnel reflection loss at air-material interfaces
- Dielectric absorption loss through wall material
- Phase delay due to slower propagation in material

Physics:
    When a 77 GHz radar signal hits a wall, three things happen:
    1. Partial reflection at each air-wall boundary (Fresnel equations)
    2. Exponential attenuation inside the wall material
    3. Phase delay because the wave travels slower (v = c/sqrt(epsilon_r))

    For the round-trip radar path (TX -> wall -> body -> wall -> RX),
    all losses and delays are doubled.

Reference: ITU-R P.2040-3, "Effects of building materials and structures
on radiowave propagation above about 100 MHz"
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple

C0 = 299792458.0  # Speed of light (m/s)


@dataclass
class WallMaterial:
    """RF material properties for wall penetration modeling.

    Attributes:
        name: Human-readable material name
        epsilon_r: Relative permittivity (dielectric constant)
        loss_tangent: Dielectric loss tangent (tan delta)
    """
    name: str
    epsilon_r: float
    loss_tangent: float


# Typical material properties at mm-wave frequencies (60-80 GHz)
# Sources: ITU-R P.2040, IEEE literature on 77 GHz automotive radar
MATERIAL_PRESETS: Dict[str, WallMaterial] = {
    'drywall':      WallMaterial('Drywall/Gypsum',  epsilon_r=2.4, loss_tangent=0.02),
    'concrete':     WallMaterial('Concrete',         epsilon_r=5.3, loss_tangent=0.05),
    'wood':         WallMaterial('Wood',             epsilon_r=2.0, loss_tangent=0.03),
    'glass':        WallMaterial('Glass',            epsilon_r=6.3, loss_tangent=0.005),
    'brick':        WallMaterial('Brick',            epsilon_r=3.7, loss_tangent=0.02),
    'plasterboard': WallMaterial('Plasterboard',     epsilon_r=2.7, loss_tangent=0.015),
}


@dataclass
class Wall:
    """A planar wall slab for through-wall radar simulation.

    The wall is an infinite plane with finite thickness, defined by
    a center point, a normal vector (pointing toward the sensor side),
    and thickness.

    Attributes:
        position: (3,) point on the wall center plane in world coords
        normal: (3,) outward normal vector (toward the sensor side)
        thickness: Wall thickness in meters
        material: WallMaterial with dielectric properties
        frequency: Operating frequency in Hz (default 77 GHz)
    """
    position: np.ndarray
    normal: np.ndarray
    thickness: float
    material: WallMaterial
    frequency: float = 77e9

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        norm = np.linalg.norm(self.normal)
        if norm < 1e-10:
            raise ValueError("Wall normal vector must be non-zero")
        self.normal = self.normal / norm

        # Pre-compute transmission parameters
        self._one_way_power, self._one_way_extra_path = self._compute_transmission()

    def _compute_transmission(self) -> Tuple[float, float]:
        """Compute one-way transmission parameters through the wall.

        Returns:
            power_factor: One-way power transmission factor (0 to 1)
            extra_path_m: Additional equivalent free-space path in meters
                          due to slower propagation inside the wall
        """
        er = self.material.epsilon_r
        tand = self.material.loss_tangent
        d = self.thickness
        f = self.frequency
        sqrt_er = np.sqrt(er)
        omega = 2 * np.pi * f

        # --- Fresnel reflection at normal incidence ---
        # Reflection coeff: Gamma = (1 - sqrt(er)) / (1 + sqrt(er))
        gamma = (1 - sqrt_er) / (1 + sqrt_er)
        T_one_interface = 1 - gamma ** 2   # power transmission per interface
        T_interfaces = T_one_interface ** 2  # entry + exit interfaces

        # --- Material absorption ---
        # Attenuation constant alpha (Np/m) for low-loss dielectric:
        #   alpha = omega * sqrt(er) * tan(delta) / (2 * c)
        alpha = omega * sqrt_er * tand / (2 * C0)
        # Power attenuation: exp(-2*alpha*d) because power ~ |E|^2
        T_material = np.exp(-2 * alpha * d)

        power_factor = T_interfaces * T_material

        # --- Phase delay ---
        # In-material phase velocity: v = c / sqrt(er)
        # Extra time vs free space: dt = d/v - d/c = d*(sqrt(er) - 1)/c
        # Equivalent extra free-space path: c * dt = d*(sqrt(er) - 1)
        extra_path_m = d * (sqrt_er - 1)

        return float(power_factor), float(extra_path_m)

    @property
    def one_way_power_transmission(self) -> float:
        """One-way power transmission factor (0 to 1)."""
        return self._one_way_power

    @property
    def one_way_extra_path(self) -> float:
        """One-way extra equivalent free-space path in meters."""
        return self._one_way_extra_path

    @property
    def round_trip_power_transmission(self) -> float:
        """Round-trip (TX -> wall -> target -> wall -> RX) power transmission."""
        return self._one_way_power ** 2

    @property
    def round_trip_extra_path(self) -> float:
        """Round-trip extra equivalent free-space path in meters."""
        return self._one_way_extra_path * 2

    @property
    def one_way_loss_dB(self) -> float:
        """One-way penetration loss in dB."""
        return -10 * np.log10(max(self._one_way_power, 1e-30))

    @property
    def round_trip_loss_dB(self) -> float:
        """Round-trip penetration loss in dB."""
        return -10 * np.log10(max(self._one_way_power ** 2, 1e-30))

    def check_rays_through_wall(self, sensor_pos: np.ndarray,
                                target_points: np.ndarray) -> np.ndarray:
        """Determine which rays from sensor to targets pass through the wall.

        A ray passes through the wall if the sensor and target are on
        opposite sides of the wall slab.

        Args:
            sensor_pos: (3,) sensor/radar position in world coordinates
            target_points: (N, 3) target reflection points

        Returns:
            through_wall: (N,) boolean mask, True if ray passes through wall
        """
        sensor_pos = np.asarray(sensor_pos, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        if target_points.ndim == 1:
            target_points = target_points.reshape(1, 3)

        # Signed distance from wall center plane
        # Positive = same side as normal vector (sensor side)
        sensor_d = np.dot(sensor_pos - self.position, self.normal)
        target_d = (target_points - self.position) @ self.normal  # (N,)

        # Ray passes through if sensor and target are on opposite sides
        through_wall = (sensor_d > 0) & (target_d < 0)
        through_wall |= (sensor_d < 0) & (target_d > 0)

        return through_wall

    def apply_to_interpolated(self, intensity: torch.Tensor,
                              pointclouds: torch.Tensor,
                              sensor_origin) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply wall attenuation and phase delay to interpolated signal data.

        For through-wall points:
        - Intensity is multiplied by sqrt(round_trip_power_transmission)
          because intensity ~ amplitude, and power ~ amplitude^2
        - Point positions are shifted away from sensor to account for
          the additional propagation delay through the wall material.
          This naturally increases the Time-of-Flight in radar computation.

        Args:
            intensity: (N,) reflection intensities
            pointclouds: (N, 3) reflection point positions in world coords
            sensor_origin: (3,) or list/tuple, sensor position

        Returns:
            (modified_intensity, modified_pointclouds)
        """
        sensor_np = np.asarray(sensor_origin, dtype=np.float64)
        pc_np = pointclouds.detach().cpu().numpy()

        through_wall = self.check_rays_through_wall(sensor_np, pc_np)
        if not through_wall.any():
            return intensity, pointclouds

        through_mask = torch.tensor(through_wall, device=intensity.device)

        # --- Amplitude attenuation ---
        # Round-trip amplitude factor = sqrt(round_trip_power)
        rt_amp = np.sqrt(self.round_trip_power_transmission)
        mod_intensity = intensity.clone()
        mod_intensity[through_mask] *= rt_amp

        # --- Phase delay via position shift ---
        # Shift through-wall points away from sensor by half the round-trip
        # extra path. This way, TX->point + point->RX distance both increase,
        # giving the correct total round-trip delay.
        mod_pc = pointclouds.clone()
        shift = self.round_trip_extra_path / 2.0

        sensor_t = torch.tensor(sensor_np, device=pointclouds.device,
                                dtype=pointclouds.dtype)
        # Direction from sensor to each point (normalize)
        dirs = mod_pc[through_mask] - sensor_t.unsqueeze(0)
        norms = dirs.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dirs = dirs / norms
        mod_pc[through_mask] += dirs * shift

        return mod_intensity, mod_pc

    def summary(self) -> str:
        """Return a human-readable summary of wall properties."""
        return (
            f"Wall: {self.material.name}, "
            f"thickness={self.thickness*100:.1f}cm, "
            f"er={self.material.epsilon_r}, "
            f"tan_d={self.material.loss_tangent}, "
            f"one-way loss={self.one_way_loss_dB:.1f}dB, "
            f"round-trip loss={self.round_trip_loss_dB:.1f}dB, "
            f"round-trip extra path={self.round_trip_extra_path*100:.1f}cm"
        )

    def __repr__(self):
        return (f"Wall(material={self.material.name}, "
                f"thickness={self.thickness*100:.1f}cm, "
                f"one_way_loss={self.one_way_loss_dB:.1f}dB)")
