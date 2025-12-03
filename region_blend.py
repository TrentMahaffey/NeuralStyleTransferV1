"""
region_blend.py — Spatial region-based model blending for neural style transfer.

Generates masks that divide a frame into regions (voronoi, fractal, diagonal, etc.)
and assigns different style models to each region, with soft feathered edges.

Also supports animated/morphing masks with organic blob/tentacle motion.
"""

import torch
import torch.nn.functional as F
import random
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass, field

# Type aliases
Mask = torch.Tensor  # Shape: [1, 1, H, W]
RegionMode = Literal["grid", "diagonal", "voronoi", "fractal", "radial", "waves", "spiral", "random"]
Assignment = Literal["sequential", "random", "weighted"]


def rotate_mask(mask: Mask, angle_degrees: float) -> Mask:
    """Rotate a mask around its center by the given angle (in degrees)."""
    if angle_degrees == 0:
        return mask

    _, _, H, W = mask.shape
    center = (W / 2, H / 2)

    # Create rotation matrix with scale factor to ensure coverage
    # Scale up slightly so rotated corners still cover the frame
    scale = 1.0

    M = cv2.getRotationMatrix2D(center, angle_degrees, scale)

    # Convert to numpy, rotate, convert back
    # Use BORDER_REPLICATE to extend edge pixels instead of black
    mask_np = mask.squeeze().numpy()
    rotated_np = cv2.warpAffine(mask_np, M, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    return torch.from_numpy(rotated_np).unsqueeze(0).unsqueeze(0)


def rotate_all_masks(masks: List[Mask], angle_degrees: float) -> List[Mask]:
    """Rotate all masks by the same angle, ensuring full frame coverage."""
    if angle_degrees == 0:
        return masks

    rotated = [rotate_mask(m, angle_degrees) for m in masks]

    # Ensure full coverage: normalize so masks sum to 1.0 everywhere
    # This prevents black areas where rotation leaves gaps
    mask_sum = torch.zeros_like(rotated[0])
    for m in rotated:
        mask_sum += m

    # Avoid division by zero and normalize
    mask_sum = mask_sum.clamp(min=1e-6)
    normalized = [m / mask_sum for m in rotated]

    return normalized


def gaussian_blur_mask(mask: Mask, sigma: float) -> Mask:
    """Apply Gaussian blur to soften mask edges."""
    if sigma <= 0:
        return mask

    # Kernel size should be odd and large enough for sigma
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel
    kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    # Apply convolution with padding
    pad = kernel_size // 2
    mask_padded = F.pad(mask, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(mask_padded, kernel_2d)

    return blurred


def feather_mask(mask: Mask, feather_px: int) -> Mask:
    """Feather mask edges using Gaussian blur."""
    if feather_px <= 0:
        return mask
    sigma = feather_px / 3.0  # 3-sigma rule
    return gaussian_blur_mask(mask, sigma)


# =============================================================================
# Pattern Generators
# =============================================================================

def grid_masks(H: int, W: int, count: int, feather: int = 20) -> List[Mask]:
    """
    Simple grid pattern. Count determines grid size:
    4 → 2x2, 9 → 3x3, 16 → 4x4, etc.
    """
    grid_size = int(math.ceil(math.sqrt(count)))
    masks = []

    cell_h = H / grid_size
    cell_w = W / grid_size

    for i in range(count):
        row = i // grid_size
        col = i % grid_size

        mask = torch.zeros(1, 1, H, W)
        y1, y2 = int(row * cell_h), int((row + 1) * cell_h)
        x1, x2 = int(col * cell_w), int((col + 1) * cell_w)

        # Clamp to bounds
        y2 = min(y2, H)
        x2 = min(x2, W)

        mask[:, :, y1:y2, x1:x2] = 1.0
        masks.append(feather_mask(mask, feather))

    return masks


def diagonal_masks(H: int, W: int, count: int, feather: int = 20,
                   rng: Optional[random.Random] = None) -> List[Mask]:
    """
    Diagonal stripe pattern. Stripes run from top-left to bottom-right
    or top-right to bottom-left (randomized).
    """
    if rng is None:
        rng = random.Random()

    masks = []

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1).expand(1, 1, H, W)
    x_coords = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W).expand(1, 1, H, W)

    # Randomly choose diagonal direction
    if rng.random() > 0.5:
        # Top-left to bottom-right
        diagonal = x_coords + y_coords
    else:
        # Top-right to bottom-left
        diagonal = (W - 1 - x_coords) + y_coords

    # Normalize to [0, 1]
    diagonal = diagonal / diagonal.max()

    # Create stripes
    for i in range(count):
        low = i / count
        high = (i + 1) / count
        mask = ((diagonal >= low) & (diagonal < high)).float()
        masks.append(feather_mask(mask, feather))

    return masks


def voronoi_masks(H: int, W: int, count: int, feather: int = 20,
                  rng: Optional[random.Random] = None,
                  region_weights: Optional[List[float]] = None) -> List[Mask]:
    """
    Voronoi pattern — balanced seed points with slight jitter for organic look.
    Each pixel belongs to nearest point. Creates roughly equal-sized cell regions
    unless region_weights is provided for variable-sized regions.

    Args:
        H, W: Frame dimensions
        count: Number of regions
        feather: Edge softness in pixels
        rng: Random number generator
        region_weights: Optional list of relative region sizes (e.g., [1,1,1,0.2] makes
                       last region ~5x smaller). Weights are normalized internally.
    """
    if rng is None:
        rng = random.Random()

    # Generate balanced seed points in a grid pattern with jitter
    # This ensures roughly equal-sized regions
    points = _generate_balanced_points(W, H, count, rng, jitter_factor=0.3)

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    x_coords = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)

    # Calculate distance to each seed point
    # Use weighted/power Voronoi if weights provided
    distances = []

    if region_weights:
        # Normalize weights so they sum to count (average weight = 1.0)
        total_weight = sum(region_weights)
        norm_weights = [w * count / total_weight for w in region_weights]

        # Larger weight = larger region = distances are scaled DOWN (closer)
        # We use: effective_dist = dist / sqrt(weight)
        # This makes higher-weighted regions "pull" more pixels
        for i, (px, py) in enumerate(points):
            dist = torch.sqrt((x_coords - px)**2 + (y_coords - py)**2)
            weight = norm_weights[i] if i < len(norm_weights) else 1.0
            # Scale distance inversely with sqrt of weight
            # Higher weight = smaller effective distance = larger region
            effective_dist = dist / (math.sqrt(weight) + 1e-6)
            distances.append(effective_dist)
    else:
        for px, py in points:
            dist = torch.sqrt((x_coords - px)**2 + (y_coords - py)**2)
            distances.append(dist)

    distances = torch.stack(distances, dim=0)  # [count, H, W]

    # Find nearest point for each pixel
    nearest = distances.argmin(dim=0)  # [H, W]

    # Create masks
    masks = []
    for i in range(count):
        mask = (nearest == i).float().view(1, 1, H, W)
        masks.append(feather_mask(mask, feather))

    return masks


def _generate_balanced_points(W: int, H: int, count: int,
                              rng: random.Random,
                              jitter_factor: float = 0.3) -> List[Tuple[float, float]]:
    """
    Generate evenly-distributed points with slight randomness for organic look.

    Uses a grid-based approach: divides the frame into a grid and places
    one point per cell with random jitter within the cell.

    Args:
        W, H: Frame dimensions
        count: Number of points to generate
        rng: Random number generator
        jitter_factor: How much to randomize within each cell (0.0 = perfect grid, 1.0 = fully random within cell)

    Returns:
        List of (x, y) tuples
    """
    # Find grid dimensions that give us approximately 'count' cells
    aspect = W / H

    # Solve: cols * rows ≈ count, cols/rows ≈ aspect
    # cols ≈ sqrt(count * aspect), rows ≈ sqrt(count / aspect)
    cols = max(1, int(math.sqrt(count * aspect) + 0.5))
    rows = max(1, int(math.sqrt(count / aspect) + 0.5))

    # Adjust to get closer to target count
    while cols * rows < count:
        if cols / rows < aspect:
            cols += 1
        else:
            rows += 1

    # Cell dimensions
    cell_w = W / cols
    cell_h = H / rows

    points = []

    # Generate points in grid pattern with jitter
    for row in range(rows):
        for col in range(cols):
            if len(points) >= count:
                break

            # Center of this cell
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h

            # Add jitter
            jitter_x = (rng.random() - 0.5) * cell_w * jitter_factor
            jitter_y = (rng.random() - 0.5) * cell_h * jitter_factor

            px = max(0, min(W - 1, cx + jitter_x))
            py = max(0, min(H - 1, cy + jitter_y))

            points.append((px, py))

    # If we still need more points (shouldn't happen often), add some randomly
    while len(points) < count:
        points.append((rng.randint(0, W - 1), rng.randint(0, H - 1)))

    # Shuffle so grid order doesn't always map to model order
    rng.shuffle(points)

    return points[:count]


def fractal_quad_masks(H: int, W: int, count: int, feather: int = 20,
                       rng: Optional[random.Random] = None,
                       max_depth: int = 4) -> List[Mask]:
    """
    Recursive quadrant subdivision — creates irregular quad-tree like regions.
    Some quads are subdivided, others are left whole.
    """
    if rng is None:
        rng = random.Random()

    regions = []  # List of (y1, y2, x1, x2) tuples

    def subdivide(y1, y2, x1, x2, depth):
        # Stop conditions
        if len(regions) >= count:
            return
        if depth >= max_depth or (y2 - y1) < 20 or (x2 - x1) < 20:
            regions.append((y1, y2, x1, x2))
            return

        # Randomly decide to subdivide or keep whole
        if rng.random() > 0.4 and depth > 0:  # 60% chance to keep whole after depth 0
            regions.append((y1, y2, x1, x2))
            return

        # Subdivide into 4 quadrants with slight randomness in split point
        mid_y = (y1 + y2) // 2 + rng.randint(-10, 10)
        mid_x = (x1 + x2) // 2 + rng.randint(-10, 10)
        mid_y = max(y1 + 10, min(y2 - 10, mid_y))
        mid_x = max(x1 + 10, min(x2 - 10, mid_x))

        # Recurse into quadrants in random order
        quads = [
            (y1, mid_y, x1, mid_x),      # top-left
            (y1, mid_y, mid_x, x2),      # top-right
            (mid_y, y2, x1, mid_x),      # bottom-left
            (mid_y, y2, mid_x, x2),      # bottom-right
        ]
        rng.shuffle(quads)

        for q in quads:
            if len(regions) >= count:
                break
            subdivide(q[0], q[1], q[2], q[3], depth + 1)

    subdivide(0, H, 0, W, 0)

    # Trim to count
    regions = regions[:count]

    # Create masks
    masks = []
    for y1, y2, x1, x2 in regions:
        mask = torch.zeros(1, 1, H, W)
        mask[:, :, y1:y2, x1:x2] = 1.0
        masks.append(feather_mask(mask, feather))

    return masks


def radial_masks(H: int, W: int, count: int, feather: int = 20,
                 rng: Optional[random.Random] = None) -> List[Mask]:
    """
    Radial wedges emanating from center (or random point).
    Like pie slices.
    """
    if rng is None:
        rng = random.Random()

    # Random center point (biased toward center)
    cx = W // 2 + rng.randint(-W // 4, W // 4)
    cy = H // 2 + rng.randint(-H // 4, H // 4)

    # Random rotation offset
    rotation = rng.random() * 2 * math.pi

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W) - cy
    x_coords = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W) - cx

    # Calculate angle for each pixel
    angles = torch.atan2(y_coords, x_coords) + math.pi + rotation  # [0, 2*pi]
    angles = angles % (2 * math.pi)

    # Create wedge masks
    masks = []
    wedge_size = 2 * math.pi / count

    for i in range(count):
        low = i * wedge_size
        high = (i + 1) * wedge_size
        mask = ((angles >= low) & (angles < high)).float().view(1, 1, H, W)
        masks.append(feather_mask(mask, feather))

    return masks


def wave_masks(H: int, W: int, count: int, feather: int = 20,
               rng: Optional[random.Random] = None) -> List[Mask]:
    """
    Wavy bands — sinusoidal boundaries between regions.
    """
    if rng is None:
        rng = random.Random()

    # Random wave parameters
    frequency = rng.uniform(1.5, 4.0)  # Number of wave cycles
    amplitude = rng.uniform(0.05, 0.15)  # Wave amplitude relative to dimension
    direction = rng.choice(['horizontal', 'vertical', 'diagonal'])
    phase = rng.random() * 2 * math.pi

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W) / H
    x_coords = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W) / W

    if direction == 'horizontal':
        # Waves run horizontally, bands stack vertically
        wave = torch.sin(x_coords * frequency * 2 * math.pi + phase) * amplitude
        position = y_coords + wave
    elif direction == 'vertical':
        # Waves run vertically, bands stack horizontally
        wave = torch.sin(y_coords * frequency * 2 * math.pi + phase) * amplitude
        position = x_coords + wave
    else:
        # Diagonal waves
        diagonal = (x_coords + y_coords) / 2
        wave = torch.sin(diagonal * frequency * 2 * math.pi + phase) * amplitude
        position = diagonal + wave

    # Normalize position to [0, 1]
    position = (position - position.min()) / (position.max() - position.min() + 1e-6)

    # Create band masks
    masks = []
    for i in range(count):
        low = i / count
        high = (i + 1) / count
        mask = ((position >= low) & (position < high)).float().view(1, 1, H, W)
        masks.append(feather_mask(mask, feather))

    return masks


def spiral_masks(H: int, W: int, count: int, feather: int = 20,
                 rng: Optional[random.Random] = None) -> List[Mask]:
    """
    Spiral pattern emanating from center.
    """
    if rng is None:
        rng = random.Random()

    # Center point
    cx, cy = W // 2, H // 2

    # Random parameters
    tightness = rng.uniform(2.0, 5.0)  # Spiral tightness
    rotation = rng.random() * 2 * math.pi

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W) - cy
    x_coords = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W) - cx

    # Calculate polar coordinates
    r = torch.sqrt(x_coords**2 + y_coords**2)
    theta = torch.atan2(y_coords, x_coords) + math.pi + rotation

    # Spiral formula: combine angle and radius
    spiral = (theta + r / max(H, W) * tightness * 2 * math.pi) % (2 * math.pi)
    spiral = spiral / (2 * math.pi)  # Normalize to [0, 1]

    # Create masks
    masks = []
    for i in range(count):
        low = i / count
        high = (i + 1) / count
        mask = ((spiral >= low) & (spiral < high)).float().view(1, 1, H, W)
        masks.append(feather_mask(mask, feather))

    return masks


def concentric_masks(H: int, W: int, count: int, feather: int = 20,
                     rng: Optional[random.Random] = None) -> List[Mask]:
    """
    Concentric rings from center.
    """
    if rng is None:
        rng = random.Random()

    # Random center
    cx = W // 2 + rng.randint(-W // 6, W // 6)
    cy = H // 2 + rng.randint(-H // 6, H // 6)

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W) - cy
    x_coords = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W) - cx

    # Distance from center
    r = torch.sqrt(x_coords**2 + y_coords**2)
    r = r / r.max()  # Normalize to [0, 1]

    # Create ring masks
    masks = []
    for i in range(count):
        low = i / count
        high = (i + 1) / count
        mask = ((r >= low) & (r < high)).float().view(1, 1, H, W)
        masks.append(feather_mask(mask, feather))

    return masks


# =============================================================================
# Animated/Morphing Mask Support (Tentacles, Blobs, Organic Motion)
# =============================================================================

@dataclass
class MorphAnimation:
    """Configuration for animated/morphing region boundaries."""
    enabled: bool = False
    speed: float = 1.0          # Animation speed multiplier
    amplitude: float = 0.15     # How far boundaries move (0.0-1.0 relative to frame)
    frequency: float = 3.0      # Noise frequency (higher = more detail/tentacles)
    octaves: int = 3            # Noise complexity (more = finer detail)
    mode: str = "blob"          # "blob", "tentacle", "wave", "pulse"
    seed: int = 42              # Random seed for reproducibility


def _perlin_noise_2d(shape: Tuple[int, int], res: Tuple[int, int],
                     seed: int = 0, tileable: bool = False) -> np.ndarray:
    """
    Generate 2D Perlin noise.

    Args:
        shape: Output shape (H, W)
        res: Resolution of the noise grid (lower = smoother)
        seed: Random seed
        tileable: Whether noise should tile seamlessly

    Returns:
        Noise array with values roughly in [-1, 1]
    """
    rng = np.random.default_rng(seed)

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def lerp(a, b, t):
        return a + t * (b - a)

    def grad(hash_val, x, y):
        h = hash_val & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y

    delta = (res[0] / shape[0], res[1] / shape[1])

    # Generate gradient grid
    grid_shape = (res[0] + 1, res[1] + 1) if not tileable else (res[0], res[1])
    angles = 2 * np.pi * rng.random(grid_shape)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Make tileable if requested
    if tileable:
        gradients = np.pad(gradients, ((0, 1), (0, 1), (0, 0)), mode='wrap')

    result = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            x = j * delta[1]
            y = i * delta[0]

            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1

            sx = fade(x - x0)
            sy = fade(y - y0)

            n00 = grad(int(rng.integers(0, 256)), x - x0, y - y0) if (y0 < gradients.shape[0] and x0 < gradients.shape[1]) else 0
            n10 = grad(int(rng.integers(0, 256)), x - x1, y - y0) if (y0 < gradients.shape[0] and x1 < gradients.shape[1]) else 0
            n01 = grad(int(rng.integers(0, 256)), x - x0, y - y1) if (y1 < gradients.shape[0] and x0 < gradients.shape[1]) else 0
            n11 = grad(int(rng.integers(0, 256)), x - x1, y - y1) if (y1 < gradients.shape[0] and x1 < gradients.shape[1]) else 0

            nx0 = lerp(n00, n10, sx)
            nx1 = lerp(n01, n11, sx)
            result[i, j] = lerp(nx0, nx1, sy)

    return result


def _simplex_noise_2d(H: int, W: int, frequency: float, octaves: int,
                      seed: int, time_offset: float = 0.0) -> np.ndarray:
    """
    Generate multi-octave 2D noise (simplified Perlin-like).
    Uses vectorized operations for performance.

    Args:
        H, W: Output dimensions
        frequency: Base frequency of noise
        octaves: Number of octave layers
        seed: Random seed
        time_offset: Temporal offset for animation

    Returns:
        Noise array with values in [0, 1]
    """
    rng = np.random.default_rng(seed)

    # Generate coordinate grids
    y = np.linspace(0, frequency, H)
    x = np.linspace(0, frequency, W)
    xx, yy = np.meshgrid(x, y)

    result = np.zeros((H, W), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    freq_mult = 1.0

    for octave in range(octaves):
        # Add time offset for animation
        offset_x = time_offset * (0.5 + 0.3 * octave) + rng.random() * 1000
        offset_y = time_offset * (0.3 + 0.2 * octave) + rng.random() * 1000

        # Simple vectorized noise using sin/cos combination
        # This approximates Perlin noise behavior
        noise = np.sin(xx * freq_mult + offset_x) * np.cos(yy * freq_mult + offset_y)
        noise += np.sin((xx + yy) * freq_mult * 0.7 + offset_x * 0.8) * 0.5
        noise += np.cos((xx - yy) * freq_mult * 0.5 + offset_y * 0.6) * 0.3

        result += noise * amplitude
        total_amplitude += amplitude
        amplitude *= 0.5
        freq_mult *= 2.0

    # Normalize to [0, 1]
    result = result / total_amplitude
    result = (result - result.min()) / (result.max() - result.min() + 1e-6)

    return result


def _generate_flow_field(H: int, W: int, frequency: float, seed: int,
                         time_offset: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D flow field (dx, dy) for warping.

    Returns:
        (flow_x, flow_y) - displacement fields
    """
    # Generate two noise fields for x and y displacement
    flow_x = _simplex_noise_2d(H, W, frequency, 2, seed, time_offset) * 2 - 1
    flow_y = _simplex_noise_2d(H, W, frequency, 2, seed + 1000, time_offset * 1.3) * 2 - 1

    return flow_x, flow_y


def warp_mask_organic(mask: Mask, amplitude: float, frequency: float,
                      frame_idx: int, speed: float, seed: int,
                      mode: str = "blob") -> Mask:
    """
    Warp a mask using organic noise-based displacement.

    Args:
        mask: Input mask [1, 1, H, W]
        amplitude: Displacement magnitude (relative to frame size)
        frequency: Noise frequency
        frame_idx: Current frame for animation
        speed: Animation speed
        seed: Random seed
        mode: Warp mode - "blob", "tentacle", "wave", "pulse"

    Returns:
        Warped mask [1, 1, H, W]
    """
    _, _, H, W = mask.shape
    time_offset = frame_idx * speed * 0.02

    # Generate displacement field based on mode
    if mode == "tentacle":
        # Higher frequency, more elongated distortions
        flow_x, flow_y = _generate_flow_field(H, W, frequency * 2, seed, time_offset)
        # Add directional bias for tentacle-like stretching
        y_coords = np.linspace(0, 1, H)[:, None]
        flow_y += np.sin(y_coords * np.pi * 3 + time_offset) * 0.5
    elif mode == "wave":
        # Sinusoidal waves
        y_coords = np.linspace(0, np.pi * frequency, H)[:, None]
        x_coords = np.linspace(0, np.pi * frequency, W)[None, :]
        flow_x = np.sin(y_coords + time_offset * 2) * np.ones((H, W))
        flow_y = np.cos(x_coords + time_offset * 1.5) * np.ones((H, W))
    elif mode == "pulse":
        # Radial pulsing
        cy, cx = H // 2, W // 2
        y_coords = np.arange(H)[:, None] - cy
        x_coords = np.arange(W)[None, :] - cx
        r = np.sqrt(x_coords**2 + y_coords**2) + 1e-6
        theta = np.arctan2(y_coords, x_coords)
        pulse = np.sin(r * 0.05 - time_offset * 3) * 0.5 + 0.5
        flow_x = np.cos(theta) * pulse
        flow_y = np.sin(theta) * pulse
    else:  # "blob" default
        flow_x, flow_y = _generate_flow_field(H, W, frequency, seed, time_offset)

    # Scale displacement by amplitude
    max_disp = max(H, W) * amplitude
    flow_x = flow_x * max_disp
    flow_y = flow_y * max_disp

    # Create remap coordinates
    y_coords = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
    x_coords = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)

    map_x = (x_coords + flow_x).astype(np.float32)
    map_y = (y_coords + flow_y).astype(np.float32)

    # Apply warp using OpenCV remap
    mask_np = mask.squeeze().numpy()
    warped = cv2.remap(mask_np, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    return torch.from_numpy(warped).unsqueeze(0).unsqueeze(0)


def warp_all_masks_organic(masks: List[Mask], morph: MorphAnimation,
                           frame_idx: int) -> List[Mask]:
    """
    Apply organic warping to all masks consistently.

    Args:
        masks: List of region masks
        morph: Morphing animation parameters
        frame_idx: Current frame number

    Returns:
        List of warped masks (normalized to sum to 1)
    """
    if not morph.enabled:
        return masks

    warped = []
    for i, mask in enumerate(masks):
        # Each region gets a slightly different seed for varied motion
        region_seed = morph.seed + i * 100
        warped_mask = warp_mask_organic(
            mask=mask,
            amplitude=morph.amplitude,
            frequency=morph.frequency,
            frame_idx=frame_idx,
            speed=morph.speed,
            seed=region_seed,
            mode=morph.mode
        )
        warped.append(warped_mask)

    # Normalize so masks sum to 1 at each pixel (prevents gaps/overlaps)
    mask_sum = torch.zeros_like(warped[0])
    for m in warped:
        mask_sum += m

    # Fix black artifacts: detect pixels with very low coverage and fill gaps
    # This happens when warping moves all masks away from certain areas
    MIN_COVERAGE = 0.1  # Threshold below which we consider it a "gap"
    gap_mask = (mask_sum < MIN_COVERAGE).float()

    if gap_mask.sum() > 0:
        # Fill gaps by dilating each mask to cover uncovered pixels
        # Use iterative dilation with increasing kernel sizes for large gaps
        filled_masks = list(warped)  # Copy list

        for kernel_size in [5, 11, 21, 41]:  # Progressively larger kernels
            padding = kernel_size // 2
            new_filled = []
            for m in filled_masks:
                # Dilate mask using max pooling with padding
                dilated = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=padding)
                # Blend original with dilated only in gap regions
                filled = m * (1 - gap_mask) + dilated * gap_mask
                new_filled.append(filled)

            # Recalculate sum and remaining gaps
            mask_sum = torch.zeros_like(new_filled[0])
            for m in new_filled:
                mask_sum += m
            gap_mask = (mask_sum < MIN_COVERAGE).float()
            filled_masks = new_filled

            # Stop early if all gaps are filled
            if gap_mask.sum() == 0:
                break

        warped = filled_masks

    # Final normalization with safe minimum
    mask_sum = mask_sum.clamp(min=1e-6)
    normalized = [m / mask_sum for m in warped]

    return normalized


def parse_morph_animation(spec: Optional[str]) -> MorphAnimation:
    """
    Parse a morph animation specification string.

    Format: "speed,amplitude,frequency,mode" or just "mode" for defaults
    Examples:
        "blob" - blob mode with defaults
        "tentacle" - tentacle mode
        "1.5,0.2,4.0,blob" - custom speed, amplitude, frequency, blob mode
        "2.0,0.1,3.0,tentacle" - faster, smaller tentacles

    Args:
        spec: Morph specification string, or None/empty for disabled

    Returns:
        MorphAnimation object
    """
    if not spec or spec.lower() in ("none", "off", "0", "static"):
        return MorphAnimation(enabled=False)

    # Check if it's just a mode name
    modes = ["blob", "tentacle", "wave", "pulse"]
    if spec.lower() in modes:
        return MorphAnimation(enabled=True, mode=spec.lower())

    parts = spec.split(",")

    try:
        if len(parts) >= 4:
            return MorphAnimation(
                enabled=True,
                speed=float(parts[0].strip()),
                amplitude=float(parts[1].strip()),
                frequency=float(parts[2].strip()),
                mode=parts[3].strip().lower()
            )
        elif len(parts) == 3:
            return MorphAnimation(
                enabled=True,
                speed=float(parts[0].strip()),
                amplitude=float(parts[1].strip()),
                frequency=float(parts[2].strip())
            )
        elif len(parts) == 2:
            return MorphAnimation(
                enabled=True,
                speed=float(parts[0].strip()),
                amplitude=float(parts[1].strip())
            )
        elif len(parts) == 1:
            # Try as speed value
            return MorphAnimation(
                enabled=True,
                speed=float(parts[0].strip())
            )
    except ValueError:
        # If parsing fails, treat as mode name
        return MorphAnimation(enabled=True, mode=spec.lower())

    return MorphAnimation(enabled=False)


# =============================================================================
# Main Interface
# =============================================================================

AVAILABLE_MODES = ["grid", "diagonal", "voronoi", "fractal", "radial",
                   "waves", "spiral", "concentric", "random"]

AVAILABLE_MORPH_MODES = ["blob", "tentacle", "wave", "pulse"]


def parse_region_sizes(spec: Optional[str], num_regions: int) -> Optional[List[float]]:
    """
    Parse region size weights specification.

    Format: "1,1,1,0.2" or "1|1|1|0.2" - relative sizes for each region
    Examples:
        "1,1,1,0.2" - 4 regions, last one is ~5x smaller
        "1,1,1,1,0.1" - 5 regions, last one is ~10x smaller (about 5% of frame)
        "2,1,1,1" - first region is 2x larger than others

    Args:
        spec: Size specification string
        num_regions: Number of regions

    Returns:
        List of relative size weights, or None if no spec
    """
    if not spec:
        return None

    # Support both comma and pipe separators
    spec = spec.replace("|", ",")
    parts = [p.strip() for p in spec.split(",") if p.strip()]

    try:
        weights = [float(p) for p in parts]
    except ValueError:
        return None

    # Extend or truncate to match num_regions
    if len(weights) < num_regions:
        # Cycle through weights
        full_weights = []
        for i in range(num_regions):
            full_weights.append(weights[i % len(weights)])
        return full_weights

    return weights[:num_regions]


def generate_region_masks(H: int, W: int, mode: RegionMode, count: int,
                          seed: Optional[int] = None,
                          feather: int = 20,
                          region_sizes: Optional[List[float]] = None) -> List[Mask]:
    """
    Generate region masks for spatial model blending.

    Args:
        H, W: Frame dimensions
        mode: Pattern type (grid, diagonal, voronoi, fractal, radial, waves, spiral, concentric, random)
        count: Number of regions to create
        seed: Random seed for reproducibility (None for random)
        feather: Edge softness in pixels
        region_sizes: Optional list of relative region sizes (e.g., [1,1,1,0.2] makes
                     last region ~5x smaller). Only works with voronoi mode currently.

    Returns:
        List of [1, 1, H, W] mask tensors, one per region
    """
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)

    if mode == "random":
        mode = rng.choice([m for m in AVAILABLE_MODES if m != "random"])
        print(f"[region] Randomly selected mode: {mode}")

    # Voronoi supports weighted regions
    if mode == "voronoi" and region_sizes:
        masks = voronoi_masks(H, W, count, feather, rng, region_sizes)
    else:
        generators = {
            "grid": lambda: grid_masks(H, W, count, feather),
            "diagonal": lambda: diagonal_masks(H, W, count, feather, rng),
            "voronoi": lambda: voronoi_masks(H, W, count, feather, rng),
            "fractal": lambda: fractal_quad_masks(H, W, count, feather, rng),
            "radial": lambda: radial_masks(H, W, count, feather, rng),
            "waves": lambda: wave_masks(H, W, count, feather, rng),
            "spiral": lambda: spiral_masks(H, W, count, feather, rng),
            "concentric": lambda: concentric_masks(H, W, count, feather, rng),
        }

        if mode not in generators:
            raise ValueError(f"Unknown region mode: {mode}. Available: {AVAILABLE_MODES}")

        masks = generators[mode]()

        if region_sizes and mode != "voronoi":
            print(f"[region] Warning: --region_sizes only works with voronoi mode, ignoring for {mode}")

    # Ensure we have exactly 'count' masks (some generators may produce fewer)
    while len(masks) < count:
        masks.append(masks[-1].clone() if masks else torch.ones(1, 1, H, W))

    return masks[:count]


def assign_models_to_regions(num_regions: int, num_models: int,
                             assignment: Assignment = "random",
                             weights: Optional[List[float]] = None,
                             seed: Optional[int] = None,
                             original_chance: float = 0.0) -> List[int]:
    """
    Assign model indices to regions.

    Args:
        num_regions: Number of regions
        num_models: Number of available models (A=0, B=1, C=2, D=3)
        assignment: "sequential" (A,B,C,D,A,B,...), "random", or "weighted"
        weights: Model weights for weighted assignment
        seed: Random seed
        original_chance: Probability (0.0-1.0) that a region stays unstyled (original).
                        Use -1 to indicate "original" as a model option in the pool.

    Returns:
        List of model indices, one per region. Index -1 means "use original frame".
    """
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)

    assignments = []

    if assignment == "sequential":
        # Include original as an option if original_chance > 0
        if original_chance > 0:
            # Treat original as model index -1, interleaved
            options = list(range(num_models)) + [-1]
            assignments = [options[i % len(options)] for i in range(num_regions)]
        else:
            assignments = [i % num_models for i in range(num_regions)]

    elif assignment == "random":
        for _ in range(num_regions):
            if original_chance > 0 and rng.random() < original_chance:
                assignments.append(-1)  # Original
            else:
                assignments.append(rng.randint(0, num_models - 1))

    elif assignment == "weighted":
        if weights is None:
            weights = [1.0 / num_models] * num_models

        # Add original to the weight pool if original_chance > 0
        if original_chance > 0:
            # Normalize existing weights to (1 - original_chance)
            total = sum(weights[:num_models])
            scaled_weights = [(w / total) * (1.0 - original_chance) for w in weights[:num_models]]
            scaled_weights.append(original_chance)  # Original gets its own weight
            options = list(range(num_models)) + [-1]
            assignments = rng.choices(options, weights=scaled_weights, k=num_regions)
        else:
            total = sum(weights[:num_models])
            weights = [w / total for w in weights[:num_models]]
            assignments = rng.choices(range(num_models), weights=weights, k=num_regions)

    else:
        raise ValueError(f"Unknown assignment mode: {assignment}")

    return assignments


def composite_regions(styled_outputs: List[torch.Tensor],
                      masks: List[Mask],
                      assignments: List[int],
                      original: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Composite styled outputs using region masks.

    Args:
        styled_outputs: List of full-frame styled tensors [C, H, W] or [1, C, H, W]
        masks: List of [1, 1, H, W] soft masks
        assignments: Model index for each mask. -1 means use original frame.
        original: Original (unstyled) frame tensor, required if any assignment is -1.

    Returns:
        Composited frame [C, H, W]
    """
    # Normalize tensor shapes
    outputs = []
    for out in styled_outputs:
        if out.dim() == 3:
            out = out.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        outputs.append(out)

    _, C, H, W = outputs[0].shape

    # Normalize original if provided
    if original is not None:
        if original.dim() == 3:
            original = original.unsqueeze(0)
        if original.shape[-2:] != (H, W):
            original = F.interpolate(original, size=(H, W), mode='bilinear', align_corners=False)

    # Accumulate weighted outputs and mask weights
    result = torch.zeros(1, C, H, W)
    weight_sum = torch.zeros(1, 1, H, W)

    for mask, model_idx in zip(masks, assignments):
        # Ensure mask matches frame size
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        # Expand mask to cover all channels
        mask_expanded = mask.expand(1, C, H, W)

        # Select source: original (-1) or styled output
        if model_idx == -1:
            if original is None:
                raise ValueError("Assignment includes original (-1) but no original frame provided")
            source = original
        else:
            source = outputs[model_idx]

        result += source * mask_expanded
        weight_sum += mask

    # Normalize by total weight (handles overlapping soft edges)
    weight_sum = weight_sum.expand(1, C, H, W).clamp(min=1e-6)
    result = result / weight_sum

    return result.squeeze(0).clamp(0, 1)  # [C, H, W]


# =============================================================================
# Enhanced region assignment with blends and scales
# =============================================================================

@dataclass
class RegionConfig:
    """Configuration for a single region."""
    model_indices: List[int]  # List of model indices to blend (-1 = original)
    model_weights: List[float]  # Blend weights for each model (must sum to 1.0)
    scale: float  # Scale factor (1.0 = full res, 0.5 = half res, etc.)


# =============================================================================
# Animated blend weights - harmonic oscillation over time
# =============================================================================

def compute_harmonic_value(
    frame_idx: int,
    period: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
    phase_offset: float = 0.0,
    waveform: str = "sine"
) -> float:
    """
    Compute a harmonic oscillation value for a given frame.

    Args:
        frame_idx: Current frame number
        period: Number of frames for one complete cycle
        min_val: Minimum output value
        max_val: Maximum output value
        phase_offset: Phase offset in degrees (0-360)
        waveform: Type of wave - "sine", "triangle", "sawtooth", "square"

    Returns:
        Value between min_val and max_val
    """
    if period <= 0:
        return (min_val + max_val) / 2

    # Normalize to 0-1 cycle position with phase offset
    t = (frame_idx / period) + (phase_offset / 360.0)
    t = t % 1.0  # Keep in 0-1 range

    # Compute waveform value (0-1 range)
    if waveform == "sine":
        # Sine wave: smooth oscillation
        wave = (math.sin(2 * math.pi * t) + 1) / 2
    elif waveform == "triangle":
        # Triangle wave: linear up then down
        if t < 0.5:
            wave = t * 2
        else:
            wave = 2 - (t * 2)
    elif waveform == "sawtooth":
        # Sawtooth: linear ramp up, instant reset
        wave = t
    elif waveform == "sawtooth_down":
        # Reverse sawtooth: instant up, linear ramp down
        wave = 1 - t
    elif waveform == "square":
        # Square wave: instant switch at 50%
        wave = 1.0 if t < 0.5 else 0.0
    else:
        # Default to sine
        wave = (math.sin(2 * math.pi * t) + 1) / 2

    # Scale to min/max range
    return min_val + wave * (max_val - min_val)


@dataclass
class BlendAnimation:
    """Animation parameters for a region's blend weights."""
    enabled: bool = False
    period: float = 120.0  # Frames per cycle
    min_opacity: float = 0.0
    max_opacity: float = 1.0
    phase_offset: float = 0.0  # Degrees
    waveform: str = "sine"
    per_model_phase: bool = True  # If True, each model gets phase-shifted


def compute_animated_weights(
    base_weights: List[float],
    frame_idx: int,
    animation: BlendAnimation
) -> List[float]:
    """
    Compute animated blend weights for a frame.

    For N models, each model's weight oscillates with a phase offset of 360/N degrees,
    creating a smooth transition effect where models fade in/out in sequence.

    Args:
        base_weights: Original static weights (used as relative strength)
        frame_idx: Current frame number
        animation: Animation parameters

    Returns:
        Animated weights that sum to 1.0
    """
    if not animation.enabled or len(base_weights) <= 1:
        return base_weights

    n = len(base_weights)
    raw_weights = []

    for i, base_w in enumerate(base_weights):
        if animation.per_model_phase:
            # Each model gets evenly distributed phase offset
            model_phase = animation.phase_offset + (i * 360.0 / n)
        else:
            # All models use same phase
            model_phase = animation.phase_offset

        # Compute oscillating value
        osc = compute_harmonic_value(
            frame_idx=frame_idx,
            period=animation.period,
            min_val=animation.min_opacity,
            max_val=animation.max_opacity,
            phase_offset=model_phase,
            waveform=animation.waveform
        )

        # Multiply by base weight to preserve relative importance
        raw_weights.append(osc * base_w)

    # Normalize to sum to 1.0
    total = sum(raw_weights)
    if total < 1e-6:
        # Avoid division by zero - fall back to equal weights
        return [1.0 / n] * n

    return [w / total for w in raw_weights]


def parse_blend_animation(spec: Optional[str]) -> BlendAnimation:
    """
    Parse a blend animation specification string.

    Format: "period,waveform,phase,min,max" or just "period" for defaults
    Examples:
        "120" - 120 frame period, sine wave, defaults
        "60,triangle" - 60 frame period, triangle wave
        "90,sine,45" - 90 frames, sine, 45 degree phase offset
        "120,sine,0,0.2,0.8" - full spec with min/max opacity

    Args:
        spec: Animation specification string, or None/empty for disabled

    Returns:
        BlendAnimation object
    """
    if not spec or spec.lower() in ("none", "static", "off", "0"):
        return BlendAnimation(enabled=False)

    parts = spec.split(",")

    try:
        period = float(parts[0].strip())
    except ValueError:
        return BlendAnimation(enabled=False)

    waveform = parts[1].strip() if len(parts) > 1 else "sine"
    phase = float(parts[2].strip()) if len(parts) > 2 else 0.0
    min_op = float(parts[3].strip()) if len(parts) > 3 else 0.0
    max_op = float(parts[4].strip()) if len(parts) > 4 else 1.0

    return BlendAnimation(
        enabled=True,
        period=period,
        min_opacity=min_op,
        max_opacity=max_op,
        phase_offset=phase,
        waveform=waveform,
        per_model_phase=True
    )


def parse_region_blend_animations(
    spec: Optional[str],
    num_regions: int
) -> List[BlendAnimation]:
    """
    Parse per-region blend animation specs.

    Format: "region0_spec|region1_spec|..." or single spec for all regions
    Examples:
        "120,sine" - same animation for all regions
        "120,sine|60,triangle|static|90,sawtooth" - per-region specs

    Args:
        spec: Animation specification string
        num_regions: Number of regions

    Returns:
        List of BlendAnimation objects, one per region
    """
    if not spec:
        return [BlendAnimation(enabled=False)] * num_regions

    # Check if it's a per-region spec (contains |) or global
    if "|" in spec:
        parts = spec.split("|")
        animations = []
        for i in range(num_regions):
            region_spec = parts[i % len(parts)].strip()
            animations.append(parse_blend_animation(region_spec))
        return animations
    else:
        # Single spec applies to all regions
        anim = parse_blend_animation(spec)
        return [anim] * num_regions


# =============================================================================
# Animated scale - resolution oscillation over time
# =============================================================================

@dataclass
class ScaleAnimation:
    """Animation parameters for a region's scale (resolution)."""
    enabled: bool = False
    period: float = 60.0        # Frames per cycle
    min_scale: float = 0.5      # Minimum scale factor
    max_scale: float = 1.0      # Maximum scale factor
    phase_offset: float = 0.0   # Degrees
    waveform: str = "sine"


def compute_animated_scale(
    base_scale: float,
    frame_idx: int,
    animation: ScaleAnimation
) -> float:
    """
    Compute animated scale for a frame.

    Args:
        base_scale: Original static scale (used as reference, animation overrides range)
        frame_idx: Current frame number
        animation: Animation parameters

    Returns:
        Animated scale value between min_scale and max_scale
    """
    if not animation.enabled:
        return base_scale

    return compute_harmonic_value(
        frame_idx=frame_idx,
        period=animation.period,
        min_val=animation.min_scale,
        max_val=animation.max_scale,
        phase_offset=animation.phase_offset,
        waveform=animation.waveform
    )


def parse_scale_animation(spec: Optional[str]) -> ScaleAnimation:
    """
    Parse a scale animation specification string.

    Format: "period,waveform,phase,min,max" or just "period" for defaults
    Examples:
        "60" - 60 frame period, sine wave, 0.5-1.0 scale
        "60,triangle" - 60 frame period, triangle wave
        "90,sine,45" - 90 frames, sine, 45 degree phase offset
        "60,sine,0,0.3,0.8" - full spec with min/max scale

    Args:
        spec: Animation specification string, or None/empty for disabled

    Returns:
        ScaleAnimation object
    """
    if not spec or spec.lower() in ("none", "static", "off", "0"):
        return ScaleAnimation(enabled=False)

    parts = spec.split(",")

    try:
        period = float(parts[0].strip())
    except ValueError:
        return ScaleAnimation(enabled=False)

    waveform = parts[1].strip() if len(parts) > 1 else "sine"
    phase = float(parts[2].strip()) if len(parts) > 2 else 0.0
    min_scale = float(parts[3].strip()) if len(parts) > 3 else 0.5
    max_scale = float(parts[4].strip()) if len(parts) > 4 else 1.0

    return ScaleAnimation(
        enabled=True,
        period=period,
        min_scale=min_scale,
        max_scale=max_scale,
        phase_offset=phase,
        waveform=waveform
    )


def parse_region_scale_animations(
    spec: Optional[str],
    num_regions: int
) -> List[ScaleAnimation]:
    """
    Parse per-region scale animation specs.

    Format: "region0_spec|region1_spec|..." or single spec for all regions
    Examples:
        "60,sine" - same animation for all regions
        "60,sine|30,triangle|static|90,sawtooth" - per-region specs

    Args:
        spec: Animation specification string
        num_regions: Number of regions

    Returns:
        List of ScaleAnimation objects, one per region
    """
    if not spec:
        return [ScaleAnimation(enabled=False)] * num_regions

    # Check if it's a per-region spec (contains |) or global
    if "|" in spec:
        parts = spec.split("|")
        animations = []
        for i in range(num_regions):
            region_spec = parts[i % len(parts)].strip()
            animations.append(parse_scale_animation(region_spec))
        return animations
    else:
        # Single spec applies to all regions
        anim = parse_scale_animation(spec)
        return [anim] * num_regions


def parse_region_configs(
    num_regions: int,
    num_models: int,
    assignment: str = "sequential",
    blend_spec: Optional[str] = None,
    scale_spec: Optional[str] = None,
    seed: Optional[int] = None,
    original_chance: float = 0.0
) -> List[RegionConfig]:
    """
    Parse region configurations including per-region blends and scales.

    Args:
        num_regions: Number of regions
        num_models: Number of available models
        assignment: Base assignment mode (sequential, random, weighted)
        blend_spec: Optional blend specification string, e.g.:
                    "A|B|C|D" = one model per region (sequential)
                    "A+B|C+D|A|B" = some regions blend models
                    "A:0.7+B:0.3|C|D" = weighted blends
        scale_spec: Optional scale specification string, e.g.:
                    "1.0,0.5,0.25,1.0" = different scales per model
                    "1.0,0.5" = alternates between scales
        seed: Random seed
        original_chance: Chance of a region being original (unblended)

    Returns:
        List of RegionConfig objects
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    configs = []

    # Parse scales if provided (supports both pipe | and comma , separators)
    scales = []
    if scale_spec:
        # Normalize to pipe separator then split
        scale_str = scale_spec.replace(",", "|")
        scales = [float(s.strip()) for s in scale_str.split("|") if s.strip()]

    # Parse blend spec if provided
    if blend_spec:
        configs = _parse_blend_spec(blend_spec, num_regions, num_models, scales, rng)
    else:
        # Use simple assignment
        assignments = assign_models_to_regions(
            num_regions, num_models, assignment, None, seed, original_chance
        )
        for i, model_idx in enumerate(assignments):
            scale = scales[i % len(scales)] if scales else 1.0
            configs.append(RegionConfig(
                model_indices=[model_idx],
                model_weights=[1.0],
                scale=scale
            ))

    return configs


def _parse_blend_spec(
    spec: str,
    num_regions: int,
    num_models: int,
    scales: List[float],
    rng: random.Random
) -> List[RegionConfig]:
    """
    Parse a blend specification string.

    Format examples:
        "A|B|C|D" = A, B, C, D in sequence
        "A+B|C+D" = blend A+B (50/50), blend C+D (50/50)
        "A:0.7+B:0.3|C" = blend A (70%) + B (30%), then C alone
        "A+B+C|D|O" = blend A+B+C, then D, then Original
    """
    model_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "O": -1, "ORIGINAL": -1}

    # Split by | to get region specs
    region_specs = [s.strip() for s in spec.upper().split("|") if s.strip()]

    configs = []

    for i in range(num_regions):
        region_spec = region_specs[i % len(region_specs)]
        scale = scales[i % len(scales)] if scales else 1.0

        # Parse this region's blend
        model_indices = []
        model_weights = []

        # Split by + to get individual models
        parts = [p.strip() for p in region_spec.split("+") if p.strip()]

        for part in parts:
            if ":" in part:
                # Has explicit weight: "A:0.7"
                model_str, weight_str = part.split(":", 1)
                model_str = model_str.strip()
                weight = float(weight_str.strip())
            else:
                model_str = part
                weight = None  # Will be distributed equally

            # Map model name to index
            if model_str in model_map:
                model_idx = model_map[model_str]
            elif model_str.isdigit():
                model_idx = int(model_str)
            else:
                raise ValueError(f"Unknown model in blend spec: {model_str}")

            model_indices.append(model_idx)
            model_weights.append(weight)

        # Fill in None weights equally
        none_count = model_weights.count(None)
        if none_count > 0:
            specified_sum = sum(w for w in model_weights if w is not None)
            remaining = max(0.0, 1.0 - specified_sum)
            equal_share = remaining / none_count if none_count > 0 else 0.0
            model_weights = [w if w is not None else equal_share for w in model_weights]

        # Normalize weights to sum to 1.0
        total = sum(model_weights)
        if total > 0:
            model_weights = [w / total for w in model_weights]
        else:
            model_weights = [1.0 / len(model_indices)] * len(model_indices)

        configs.append(RegionConfig(
            model_indices=model_indices,
            model_weights=model_weights,
            scale=scale
        ))

    return configs


def composite_regions_advanced(
    styled_outputs_by_scale: dict,  # {scale: [outputs for each model]}
    masks: List[Mask],
    configs: List[RegionConfig],
    original: Optional[torch.Tensor] = None,
    H: int = 0,
    W: int = 0
) -> torch.Tensor:
    """
    Advanced compositing with per-region model blends and scales.

    Args:
        styled_outputs_by_scale: Dict mapping scale -> list of model outputs at that scale
                                 e.g., {1.0: [A_full, B_full, C_full, D_full],
                                        0.5: [A_half, B_half, C_half, D_half]}
        masks: Region masks
        configs: Per-region configuration (which models, weights, scale)
        original: Original frame (for regions with model_idx=-1)
        H, W: Output dimensions

    Returns:
        Composited frame [C, H, W]
    """
    if not styled_outputs_by_scale:
        raise ValueError("No styled outputs provided")

    # Get dimensions from first available output
    first_scale = list(styled_outputs_by_scale.keys())[0]
    first_output = styled_outputs_by_scale[first_scale][0]
    if first_output.dim() == 3:
        C = first_output.shape[0]
        if H == 0 or W == 0:
            H, W = first_output.shape[1], first_output.shape[2]
    else:
        C = first_output.shape[1]
        if H == 0 or W == 0:
            H, W = first_output.shape[2], first_output.shape[3]

    # Normalize original if provided
    if original is not None:
        if original.dim() == 3:
            original = original.unsqueeze(0)
        if original.shape[-2:] != (H, W):
            original = F.interpolate(original, size=(H, W), mode='bilinear', align_corners=False)

    # Accumulate result
    result = torch.zeros(1, C, H, W)
    weight_sum = torch.zeros(1, 1, H, W)

    for mask, config in zip(masks, configs):
        # Ensure mask matches frame size
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        mask_expanded = mask.expand(1, C, H, W)

        # Get outputs at this region's scale
        scale = config.scale
        if scale not in styled_outputs_by_scale:
            # Fall back to nearest available scale
            available_scales = list(styled_outputs_by_scale.keys())
            scale = min(available_scales, key=lambda s: abs(s - config.scale))

        outputs_at_scale = styled_outputs_by_scale[scale]

        # Blend models for this region
        region_blend = torch.zeros(1, C, H, W)
        for model_idx, weight in zip(config.model_indices, config.model_weights):
            if model_idx == -1:
                if original is None:
                    raise ValueError("Region config uses original (-1) but no original frame provided")
                source = original
            else:
                source = outputs_at_scale[model_idx]
                if source.dim() == 3:
                    source = source.unsqueeze(0)

                # Upscale if needed
                if source.shape[-2:] != (H, W):
                    source = F.interpolate(source, size=(H, W), mode='bilinear', align_corners=False)

            region_blend += weight * source

        result += region_blend * mask_expanded
        weight_sum += mask

    # Normalize
    weight_sum = weight_sum.expand(1, C, H, W).clamp(min=1e-6)
    result = result / weight_sum

    return result.squeeze(0).clamp(0, 1)


# =============================================================================
# Convenience wrapper for pipeline integration
# =============================================================================

# Cache for base masks (to avoid regenerating every frame when rotating)
_mask_cache: dict = {}


def blend_by_regions(styled_outputs: List[torch.Tensor],
                     H: int, W: int,
                     mode: RegionMode = "voronoi",
                     region_count: Optional[int] = None,
                     assignment: Assignment = "random",
                     weights: Optional[List[float]] = None,
                     feather: int = 20,
                     seed: Optional[int] = None,
                     original: Optional[torch.Tensor] = None,
                     original_chance: float = 0.0,
                     frame_idx: int = 0,
                     rotation_rate: float = 0.0,
                     morph: Optional[MorphAnimation] = None) -> torch.Tensor:
    """
    High-level function: generate masks, assign models, composite.

    Args:
        styled_outputs: List of styled frame tensors (one per model)
        H, W: Frame dimensions
        mode: Region pattern type
        region_count: Number of regions (defaults to len(styled_outputs))
        assignment: How to assign models to regions
        weights: Model weights for weighted assignment
        feather: Edge softness in pixels
        seed: Random seed for reproducibility
        original: Original (unstyled) frame tensor for regions that stay unstyled
        original_chance: Probability (0.0-1.0) that a region stays unstyled
        frame_idx: Current frame index (for rotation calculation)
        rotation_rate: Degrees to rotate per frame (e.g., 2.0 = 2° per frame)
        morph: Optional MorphAnimation for organic blob/tentacle motion

    Returns:
        Composited frame tensor [C, H, W]
    """
    global _mask_cache

    num_models = len(styled_outputs)
    if region_count is None:
        region_count = num_models

    # Determine if we need to cache masks (rotation or morph animation)
    needs_caching = rotation_rate != 0 or (morph and morph.enabled)

    # Create cache key for base masks
    cache_key = (H, W, mode, region_count, seed, feather)

    # Generate or retrieve cached base masks
    if needs_caching and cache_key in _mask_cache:
        # Use cached masks for rotation/morph
        base_masks, cached_assignments = _mask_cache[cache_key]
        assignments = cached_assignments
    else:
        # Generate new masks
        base_masks = generate_region_masks(H, W, mode, region_count, seed, feather)

        # Assign models to regions (do this once and cache with masks)
        assignments = assign_models_to_regions(
            len(base_masks), num_models, assignment, weights, seed, original_chance
        )

        # Cache if we're animating
        if needs_caching:
            _mask_cache[cache_key] = (base_masks, assignments)

    # Start with base masks
    masks = base_masks

    # Apply rotation if specified
    if rotation_rate != 0:
        angle = frame_idx * rotation_rate
        masks = rotate_all_masks(masks, angle)
        # Re-feather after rotation to smooth any aliasing
        masks = [feather_mask(m, feather // 2) for m in masks]

    # Apply organic morph animation if specified
    if morph and morph.enabled:
        masks = warp_all_masks_organic(masks, morph, frame_idx)
        # Light feathering to smooth warped edges
        masks = [feather_mask(m, max(5, feather // 4)) for m in masks]

    # If not animating, regenerate assignments each frame (original behavior)
    if not needs_caching:
        assignments = assign_models_to_regions(
            len(masks), num_models, assignment, weights, seed, original_chance
        )

    # Count how many regions use original
    num_original = sum(1 for a in assignments if a == -1)

    if frame_idx <= 2 or frame_idx % 50 == 0:
        rot_info = f" rotation={frame_idx * rotation_rate:.1f}°" if rotation_rate != 0 else ""
        morph_info = f" morph={morph.mode}" if (morph and morph.enabled) else ""
        print(f"[region] mode={mode} regions={len(masks)} models={num_models} "
              f"assignment={assignment} feather={feather}px seed={seed} "
              f"original_regions={num_original}/{len(masks)}{rot_info}{morph_info}")

    # Composite
    return composite_regions(styled_outputs, masks, assignments, original)


def clear_mask_cache():
    """Clear the cached masks (call between videos or when parameters change)."""
    global _mask_cache
    _mask_cache = {}


def get_required_scales(
    num_regions: int,
    num_models: int,
    assignment: str = "sequential",
    blend_spec: Optional[str] = None,
    scale_spec: Optional[str] = None,
    seed: Optional[int] = None,
    original_chance: float = 0.0
) -> List[float]:
    """
    Determine which scales are needed for region blending.

    Returns a list of unique scales that will be needed, so the pipeline
    can pre-generate styled outputs at each scale.
    """
    if not scale_spec:
        return [1.0]  # Only full resolution needed

    # Parse scales (supports both pipe | and comma , separators)
    scale_str = scale_spec.replace(",", "|")
    scales = [float(s.strip()) for s in scale_str.split("|") if s.strip()]
    if not scales:
        return [1.0]

    # If blend_spec is provided, configs cycle through region_specs
    if blend_spec:
        configs = _parse_blend_spec(
            blend_spec, num_regions, num_models, scales,
            random.Random(seed) if seed else random.Random()
        )
        return list(set(c.scale for c in configs))

    # Otherwise scales just cycle
    return list(set(scales))


def blend_by_regions_advanced(
    styled_outputs_by_scale: dict,  # {scale: [A_output, B_output, C_output, D_output]}
    H: int, W: int,
    mode: RegionMode = "voronoi",
    region_count: Optional[int] = None,
    assignment: Assignment = "random",
    blend_spec: Optional[str] = None,
    scale_spec: Optional[str] = None,
    weights: Optional[List[float]] = None,
    feather: int = 20,
    seed: Optional[int] = None,
    original: Optional[torch.Tensor] = None,
    original_chance: float = 0.0,
    frame_idx: int = 0,
    rotation_rate: float = 0.0,
    morph: Optional[MorphAnimation] = None
) -> torch.Tensor:
    """
    Advanced region blending with per-region model blends and scales.

    Args:
        styled_outputs_by_scale: Dict mapping scale -> list of model outputs at that scale
                                 e.g., {1.0: [A_full, B_full], 0.5: [A_half, B_half]}
        H, W: Frame dimensions
        mode: Region pattern type
        region_count: Number of regions
        assignment: Base assignment mode (sequential, random, weighted)
        blend_spec: Blend specification string, e.g., "A+B|C|D|A:0.7+B:0.3"
        scale_spec: Scale specification string, e.g., "1.0,0.5,0.25"
        weights: Model weights (used when no blend_spec)
        feather: Edge softness in pixels
        seed: Random seed
        original: Original frame for unstyled regions
        original_chance: Probability of unstyled regions
        frame_idx: Current frame index (for rotation)
        rotation_rate: Degrees to rotate per frame
        morph: Optional MorphAnimation for organic blob/tentacle motion

    Returns:
        Composited frame tensor [C, H, W]
    """
    global _mask_cache

    # Get num_models from first scale's outputs
    first_scale = list(styled_outputs_by_scale.keys())[0]
    num_models = len(styled_outputs_by_scale[first_scale])

    if region_count is None:
        region_count = num_models

    # Determine if we need to cache masks
    needs_caching = rotation_rate != 0 or (morph and morph.enabled)

    # Create cache key
    cache_key = (H, W, mode, region_count, seed, feather, blend_spec, scale_spec)

    # Generate or retrieve cached masks and configs
    if needs_caching and cache_key in _mask_cache:
        base_masks, configs = _mask_cache[cache_key]
    else:
        # Generate masks
        base_masks = generate_region_masks(H, W, mode, region_count, seed, feather)

        # Generate configs
        configs = parse_region_configs(
            num_regions=len(base_masks),
            num_models=num_models,
            assignment=assignment,
            blend_spec=blend_spec,
            scale_spec=scale_spec,
            seed=seed,
            original_chance=original_chance
        )

        # Cache if animating
        if needs_caching:
            _mask_cache[cache_key] = (base_masks, configs)

    # Start with base masks
    masks = base_masks

    # Apply rotation if specified
    if rotation_rate != 0:
        angle = frame_idx * rotation_rate
        masks = rotate_all_masks(masks, angle)
        masks = [feather_mask(m, feather // 2) for m in masks]

    # Apply organic morph animation if specified
    if morph and morph.enabled:
        masks = warp_all_masks_organic(masks, morph, frame_idx)
        masks = [feather_mask(m, max(5, feather // 4)) for m in masks]

    # If not animating, regenerate configs each frame (original behavior)
    if not needs_caching:
        configs = parse_region_configs(
            num_regions=len(masks),
            num_models=num_models,
            assignment=assignment,
            blend_spec=blend_spec,
            scale_spec=scale_spec,
            seed=seed,
            original_chance=original_chance
        )

    # Log info
    if frame_idx <= 2 or frame_idx % 50 == 0:
        rot_info = f" rotation={frame_idx * rotation_rate:.1f}°" if rotation_rate != 0 else ""
        morph_info = f" morph={morph.mode}" if (morph and morph.enabled) else ""
        scales_used = list(styled_outputs_by_scale.keys())
        print(f"[region-adv] mode={mode} regions={len(masks)} models={num_models} "
              f"scales={scales_used} blend_spec={blend_spec or 'none'}{rot_info}{morph_info}")

    # Composite using advanced method
    return composite_regions_advanced(
        styled_outputs_by_scale=styled_outputs_by_scale,
        masks=masks,
        configs=configs,
        original=original,
        H=H, W=W
    )


# =============================================================================
# Region-Optimized Styling (crop-based for efficiency)
# =============================================================================

@dataclass
class RegionCrop:
    """Information about a cropped region for optimized styling."""
    region_idx: int
    mask: Mask  # Original full-frame mask
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) - tight bounding box
    padded_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) - with padding
    crop_mask: Mask  # Mask cropped to padded_bbox
    config: RegionConfig  # Model assignment and blend info


def compute_mask_bbox(mask: Mask, threshold: float = 0.01) -> Tuple[int, int, int, int]:
    """
    Compute tight bounding box of non-zero mask region.

    Args:
        mask: [1, 1, H, W] tensor
        threshold: Values above this are considered part of the region

    Returns:
        (x1, y1, x2, y2) bounding box coordinates
    """
    _, _, H, W = mask.shape
    mask_2d = mask.squeeze().numpy()

    # Find non-zero pixels
    rows = np.any(mask_2d > threshold, axis=1)
    cols = np.any(mask_2d > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty mask - return full frame
        return (0, 0, W, H)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return (int(x1), int(y1), int(x2) + 1, int(y2) + 1)


def pad_bbox(bbox: Tuple[int, int, int, int], padding: int,
             max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    """
    Add padding to bounding box, clamping to frame boundaries.

    Args:
        bbox: (x1, y1, x2, y2)
        padding: Pixels to add on each side
        max_w, max_h: Frame dimensions

    Returns:
        Padded (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(max_w, x2 + padding),
        min(max_h, y2 + padding)
    )


def prepare_region_crops(
    masks: List[Mask],
    configs: List[RegionConfig],
    H: int, W: int,
    padding: int = 64
) -> List[RegionCrop]:
    """
    Prepare crop information for each region.

    Args:
        masks: List of region masks
        configs: Region configurations (model assignments)
        H, W: Frame dimensions
        padding: Pixels of context to include around each region

    Returns:
        List of RegionCrop objects with bbox and crop info
    """
    crops = []

    for idx, (mask, config) in enumerate(zip(masks, configs)):
        # Compute tight bounding box
        bbox = compute_mask_bbox(mask)

        # Add padding for convolution context
        padded_bbox = pad_bbox(bbox, padding, W, H)

        # Crop the mask to the padded region
        px1, py1, px2, py2 = padded_bbox
        crop_mask = mask[:, :, py1:py2, px1:px2].clone()

        crops.append(RegionCrop(
            region_idx=idx,
            mask=mask,
            bbox=bbox,
            padded_bbox=padded_bbox,
            crop_mask=crop_mask,
            config=config
        ))

    return crops


def extract_crop(frame: torch.Tensor, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Extract a crop from a frame tensor.

    Args:
        frame: [C, H, W] or [1, C, H, W] tensor
        bbox: (x1, y1, x2, y2)

    Returns:
        Cropped tensor
    """
    if frame.dim() == 4:
        frame = frame.squeeze(0)

    x1, y1, x2, y2 = bbox
    return frame[:, y1:y2, x1:x2].clone()


def place_crop(canvas: torch.Tensor, crop: torch.Tensor,
               bbox: Tuple[int, int, int, int], mask: Mask) -> None:
    """
    Place a styled crop back onto the canvas using the mask.

    Args:
        canvas: [C, H, W] accumulator tensor (modified in place)
        crop: [C, crop_H, crop_W] styled crop
        bbox: (x1, y1, x2, y2) where to place it
        mask: [1, 1, crop_H, crop_W] weight mask for this region
    """
    x1, y1, x2, y2 = bbox
    C = canvas.shape[0]

    # Ensure crop matches bbox size
    crop_h, crop_w = y2 - y1, x2 - x1
    if crop.shape[1] != crop_h or crop.shape[2] != crop_w:
        crop = F.interpolate(crop.unsqueeze(0), size=(crop_h, crop_w),
                            mode='bilinear', align_corners=False).squeeze(0)

    # Ensure mask matches crop size
    if mask.shape[2] != crop_h or mask.shape[3] != crop_w:
        mask = F.interpolate(mask, size=(crop_h, crop_w),
                            mode='bilinear', align_corners=False)

    # Expand mask to all channels
    mask_expanded = mask.squeeze(0).expand(C, -1, -1)  # [C, crop_H, crop_W]

    # Add weighted crop to canvas
    canvas[:, y1:y2, x1:x2] += crop * mask_expanded


def get_models_needed_for_regions(crops: List[RegionCrop]) -> List[int]:
    """
    Get sorted list of unique model indices needed across all regions.

    Args:
        crops: List of RegionCrop objects

    Returns:
        Sorted list of model indices (excluding -1 for original)
    """
    model_indices = set()
    for crop in crops:
        for idx in crop.config.model_indices:
            if idx >= 0:  # Skip original (-1)
                model_indices.add(idx)
    return sorted(model_indices)


def get_regions_for_model(crops: List[RegionCrop], model_idx: int) -> List[RegionCrop]:
    """
    Get all regions that need a specific model.

    Args:
        crops: List of RegionCrop objects
        model_idx: Model index to filter by

    Returns:
        List of RegionCrop objects that use this model
    """
    return [c for c in crops if model_idx in c.config.model_indices]


def merge_bboxes(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """
    Merge multiple bounding boxes into one that contains all of them.

    Args:
        bboxes: List of (x1, y1, x2, y2) tuples

    Returns:
        Single (x1, y1, x2, y2) containing all input boxes
    """
    if not bboxes:
        return (0, 0, 0, 0)

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)

    return (x1, y1, x2, y2)


def compute_crop_coverage(crops: List[RegionCrop], H: int, W: int) -> float:
    """
    Compute what percentage of the frame is covered by region crops.

    Args:
        crops: List of RegionCrop objects
        H, W: Frame dimensions

    Returns:
        Coverage ratio (0.0 to 1.0+, can exceed 1.0 with overlap)
    """
    total_pixels = H * W
    crop_pixels = 0

    for crop in crops:
        x1, y1, x2, y2 = crop.padded_bbox
        crop_pixels += (x2 - x1) * (y2 - y1)

    return crop_pixels / total_pixels


def composite_from_crops(
    styled_crops: dict,  # {model_idx: {region_idx: styled_crop_tensor}}
    crops: List[RegionCrop],
    original: Optional[torch.Tensor],
    H: int, W: int,
    frame_idx: int = 0,
    blend_animations: Optional[List[BlendAnimation]] = None
) -> torch.Tensor:
    """
    Composite final frame from individually styled crops.

    Args:
        styled_crops: Nested dict mapping model_idx -> region_idx -> styled crop tensor
        crops: List of RegionCrop objects with mask and config info
        original: Original frame for regions with model_idx=-1
        H, W: Output frame dimensions
        frame_idx: Current frame number (for animated blends)
        blend_animations: Optional list of BlendAnimation per region

    Returns:
        Composited frame [C, H, W]
    """
    C = 3  # RGB
    canvas = torch.zeros(C, H, W)
    weight_sum = torch.zeros(1, H, W)

    for crop_info in crops:
        config = crop_info.config
        x1, y1, x2, y2 = crop_info.padded_bbox
        crop_h, crop_w = y2 - y1, x2 - x1

        # Get weights - apply animation if provided
        if blend_animations and crop_info.region_idx < len(blend_animations):
            anim = blend_animations[crop_info.region_idx]
            weights = compute_animated_weights(config.model_weights, frame_idx, anim)
        else:
            weights = config.model_weights

        # Blend models for this region (within the crop)
        region_blend = torch.zeros(C, crop_h, crop_w)

        for model_idx, weight in zip(config.model_indices, weights):
            if model_idx == -1:
                # Use original
                if original is None:
                    raise ValueError("Region uses original but no original provided")
                source = extract_crop(original, crop_info.padded_bbox)
            else:
                # Use styled crop
                if model_idx not in styled_crops:
                    raise ValueError(f"Model {model_idx} not in styled_crops")
                if crop_info.region_idx not in styled_crops[model_idx]:
                    raise ValueError(f"Region {crop_info.region_idx} not styled by model {model_idx}")
                source = styled_crops[model_idx][crop_info.region_idx]

            # Ensure source matches crop size
            if source.shape[1] != crop_h or source.shape[2] != crop_w:
                source = F.interpolate(source.unsqueeze(0), size=(crop_h, crop_w),
                                      mode='bilinear', align_corners=False).squeeze(0)

            region_blend += weight * source

        # Place blended region onto canvas using mask
        place_crop(canvas, region_blend, crop_info.padded_bbox, crop_info.crop_mask)

        # Accumulate weights
        mask = crop_info.crop_mask
        if mask.shape[2] != crop_h or mask.shape[3] != crop_w:
            mask = F.interpolate(mask, size=(crop_h, crop_w),
                                mode='bilinear', align_corners=False)
        weight_sum[:, y1:y2, x1:x2] += mask.squeeze(0)

    # Normalize by total weight, handling gaps (low coverage areas)
    # Detect gaps where weight is very low - these cause black artifacts
    MIN_COVERAGE = 0.1
    gap_mask = (weight_sum < MIN_COVERAGE).float()

    # If we have gaps, fill them to prevent black artifacts
    if gap_mask.sum() > 0:
        gap_mask_expanded = gap_mask.expand(C, -1, -1)

        if original is not None:
            # Fill canvas gaps with original frame
            canvas = canvas + original * gap_mask_expanded
            weight_sum = weight_sum + gap_mask
        else:
            # No original available - fill gaps by dilating the canvas
            # This spreads nearby styled pixels into the gaps
            for kernel_size in [5, 11, 21]:
                padding = kernel_size // 2
                # Dilate canvas values (sum of weighted pixels)
                canvas_dilated = F.max_pool2d(canvas.unsqueeze(0), kernel_size=kernel_size,
                                              stride=1, padding=padding).squeeze(0)
                # Dilate weights
                weight_dilated = F.max_pool2d(weight_sum.unsqueeze(0), kernel_size=kernel_size,
                                              stride=1, padding=padding).squeeze(0)
                # Fill only gap areas
                canvas = canvas * (1 - gap_mask_expanded) + canvas_dilated * gap_mask_expanded
                weight_sum = weight_sum * (1 - gap_mask) + weight_dilated * gap_mask
                # Recalculate gaps
                gap_mask = (weight_sum < MIN_COVERAGE).float()
                gap_mask_expanded = gap_mask.expand(C, -1, -1)
                if gap_mask.sum() == 0:
                    break

    weight_sum = weight_sum.expand(C, -1, -1).clamp(min=1e-6)
    canvas = canvas / weight_sum

    return canvas.clamp(0, 1)
