import numpy as np
from skimage import measure
from scipy import ndimage

def voxels_to_smooth_mesh(voxel_grid, level=0.5, smooth_iterations=3):
    """
    Convert voxel grid to smooth triangular mesh using Marching Cubes
    This creates PROFESSIONAL QUALITY smooth surfaces
    """
    # Apply multiple smoothing passes for ultra-smooth results
    smoothed = voxel_grid.astype(float)
    for _ in range(smooth_iterations):
        smoothed = ndimage.gaussian_filter(smoothed, sigma=1.0)

    # Use marching cubes to create smooth mesh
    try:
        verts, faces, normals, values = measure.marching_cubes(
            smoothed,
            level=level,
            spacing=(1.0, 1.0, 1.0),
            gradient_direction='descent',
            step_size=1,
            allow_degenerate=False
        )

        # Further smooth the vertices
        verts = smooth_mesh_vertices(verts, faces, iterations=2)

        return verts, faces

    except Exception as e:
        print(f"Marching cubes error: {e}")
        # Fallback to simple cube mesh
        return create_fallback_mesh()


def smooth_mesh_vertices(vertices, faces, iterations=2):
    """
    Laplacian smoothing for even smoother mesh
    """
    for _ in range(iterations):
        # Build adjacency
        adjacency = {}
        for face in faces:
            for i in range(3):
                v1 = face[i]
                v2 = face[(i+1) % 3]
                if v1 not in adjacency:
                    adjacency[v1] = []
                if v2 not in adjacency:
                    adjacency[v2] = []
                if v2 not in adjacency[v1]:
                    adjacency[v1].append(v2)
                if v1 not in adjacency[v2]:
                    adjacency[v2].append(v1)

        # Smooth vertices
        new_vertices = vertices.copy()
        for i, neighbors in adjacency.items():
            if len(neighbors) > 0:
                avg = np.mean([vertices[n] for n in neighbors], axis=0)
                new_vertices[i] = 0.5 * vertices[i] + 0.5 * avg

        vertices = new_vertices

    return vertices


def create_fallback_mesh():
    """Simple cube as fallback"""
    vertices = np.array([
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10]
    ], dtype=float)

    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
    ])

    return vertices, faces


def create_parametric_chair(resolution=64):
    """
    Create HIGH-QUALITY chair using parametric surfaces
    """
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    # More detailed, realistic proportions
    seat_height = int(r * 0.5)
    seat_depth = int(r * 0.4)
    seat_width = int(r * 0.4)

    # Curved seat with soft edges
    for i in range(seat_height - 2, seat_height + 6):
        for j in range(int(r*0.25), int(r*0.25) + seat_depth):
            for k in range(int(r*0.25), int(r*0.25) + seat_width):
                # Add curvature to seat
                center_j = int(r*0.25) + seat_depth // 2
                center_k = int(r*0.25) + seat_width // 2
                dist_j = abs(j - center_j) / (seat_depth / 2)
                dist_k = abs(k - center_k) / (seat_width / 2)
                curve = 1.0 - 0.15 * (dist_j**2 + dist_k**2)
                if curve > 0:
                    voxels[i, j, k] = curve

    # Curved backrest
    back_start = int(r*0.25)
    for i in range(seat_height + 6, min(seat_height + 35, r)):
        for j in range(back_start, back_start + 6):
            for k in range(int(r*0.25), int(r*0.25) + seat_width):
                # Ergonomic curve
                height_factor = (i - seat_height - 6) / 30.0
                center_k = int(r*0.25) + seat_width // 2
                dist_k = abs(k - center_k) / (seat_width / 2)
                curve = 1.0 - 0.1 * dist_k**2 - 0.05 * height_factor**2
                if curve > 0:
                    voxels[i, j, k] = curve

    # Tapered legs with realistic proportions
    leg_positions = [
        (int(r*0.28), int(r*0.28)),
        (int(r*0.28), int(r*0.60)),
        (int(r*0.60), int(r*0.28)),
        (int(r*0.60), int(r*0.60))
    ]

    for leg_j, leg_k in leg_positions:
        for i in range(5, seat_height - 2):
            # Taper: thicker at top, thinner at bottom
            progress = (i - 5) / (seat_height - 7)
            thickness = 4 - int(progress * 1.5)

            for j in range(leg_j - thickness, leg_j + thickness):
                for k in range(leg_k - thickness, leg_k + thickness):
                    if 0 <= j < r and 0 <= k < r:
                        # Round the leg
                        dj = j - leg_j
                        dk = k - leg_k
                        dist = np.sqrt(dj**2 + dk**2)
                        if dist <= thickness:
                            voxels[i, j, k] = 1.0 - (dist / thickness) * 0.3

    return voxels


def create_parametric_table(resolution=64):
    """HIGH-QUALITY table with realistic details"""
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    table_height = int(r * 0.65)

    # Table top with rounded edges
    for i in range(table_height, table_height + 5):
        for j in range(int(r*0.1), int(r*0.9)):
            for k in range(int(r*0.1), int(r*0.9)):
                # Round the edges
                edge_dist_j = min(j - int(r*0.1), int(r*0.9) - j)
                edge_dist_k = min(k - int(r*0.1), int(r*0.9) - k)
                edge_dist = min(edge_dist_j, edge_dist_k)

                if edge_dist >= 3:
                    voxels[i, j, k] = 1.0
                elif edge_dist > 0:
                    # Rounded bevel
                    voxels[i, j, k] = edge_dist / 3.0

    # Elegant tapered legs
    leg_positions = [
        (int(r*0.15), int(r*0.15)),
        (int(r*0.15), int(r*0.85)),
        (int(r*0.85), int(r*0.15)),
        (int(r*0.85), int(r*0.85))
    ]

    for leg_j, leg_k in leg_positions:
        for i in range(5, table_height):
            progress = (i - 5) / (table_height - 5)
            thickness = 5 - int(progress * 2)

            for j in range(leg_j - thickness, leg_j + thickness):
                for k in range(leg_k - thickness, leg_k + thickness):
                    if 0 <= j < r and 0 <= k < r:
                        dj = j - leg_j
                        dk = k - leg_k
                        dist = np.sqrt(dj**2 + dk**2)
                        if dist <= thickness:
                            voxels[i, j, k] = 1.0

    return voxels


def create_parametric_bottle(resolution=64):
    """HIGH-QUALITY bottle with smooth curves"""
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution
    center = r // 2

    # Base (wider, rounded)
    for i in range(5, 18):
        progress = (i - 5) / 13.0
        radius = 12 - progress * 3
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= radius:
                    falloff = 1.0 - (dist / radius) * 0.3
                    voxels[i, j, k] = falloff

    # Body (elegant curve)
    for i in range(18, 48):
        # Slight hourglass
        progress = (i - 18) / 30.0
        radius = 9 - np.sin(progress * np.pi) * 2
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= radius:
                    voxels[i, j, k] = 1.0

    # Shoulder (smooth transition)
    for i in range(48, 54):
        progress = (i - 48) / 6.0
        radius = 9 * (1 - progress) + 4 * progress
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= radius:
                    voxels[i, j, k] = 1.0

    # Neck (straight, narrow)
    for i in range(54, 60):
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= 4:
                    voxels[i, j, k] = 1.0

    return voxels
