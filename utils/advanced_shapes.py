import numpy as np
from scipy import ndimage

def create_smooth_chair(resolution=48):
    """Create detailed chair with smooth surfaces"""
    v = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    # Randomize dimensions slightly
    seat_h = int(r * 0.55) + np.random.randint(-2, 3)
    seat_d = int(r * 0.35)
    seat_w = int(r * 0.35)
    back_h = int(r * 0.5)

    # Seat
    for i in range(seat_h, min(seat_h + 4, r)):
        for j in range(int(r*0.3), min(int(r*0.3) + seat_d, r)):
            for k in range(int(r*0.3), min(int(r*0.3) + seat_w, r)):
                v[i, j, k] = 1.0

    # Backrest
    for i in range(seat_h + 4, min(seat_h + 4 + back_h, r)):
        for j in range(int(r*0.3), min(int(r*0.35), r)):
            for k in range(int(r*0.3), min(int(r*0.3) + seat_w, r)):
                v[i, j, k] = 0.9

    # 4 Legs
    leg_thick = 3
    for leg_j, leg_k in [(int(r*0.32), int(r*0.32)), (int(r*0.32), int(r*0.62)),
                          (int(r*0.62), int(r*0.32)), (int(r*0.62), int(r*0.62))]:
        for i in range(int(r*0.1), seat_h):
            for j in range(max(0, leg_j-leg_thick//2), min(r, leg_j+leg_thick//2)):
                for k in range(max(0, leg_k-leg_thick//2), min(r, leg_k+leg_thick//2)):
                    v[i, j, k] = 1.0

    # Smooth
    v = ndimage.gaussian_filter(v, sigma=1.5)
    return v > 0.2


def create_smooth_table(resolution=48):
    """Create detailed table"""
    v = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    table_h = int(r * 0.7) + np.random.randint(-2, 3)
    top_thick = 4

    # Tabletop
    for i in range(table_h, min(table_h + top_thick, r)):
        for j in range(int(r*0.15), min(int(r*0.85), r)):
            for k in range(int(r*0.15), min(int(r*0.85), r)):
                v[i, j, k] = 1.0

    # 4 Legs
    leg_thick = 4
    for leg_j, leg_k in [(int(r*0.2), int(r*0.2)), (int(r*0.2), int(r*0.8)),
                          (int(r*0.8), int(r*0.2)), (int(r*0.8), int(r*0.8))]:
        for i in range(int(r*0.1), table_h):
            for j in range(max(0, leg_j-leg_thick//2), min(r, leg_j+leg_thick//2)):
                for k in range(max(0, leg_k-leg_thick//2), min(r, leg_k+leg_thick//2)):
                    v[i, j, k] = 1.0

    v = ndimage.gaussian_filter(v, sigma=1.2)
    return v > 0.2


def create_smooth_bottle(resolution=48):
    """Create smooth bottle"""
    v = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution
    center = r // 2

    # Randomize
    base_r = int(r * 0.25) + np.random.randint(-1, 2)
    body_r = int(r * 0.18) + np.random.randint(-1, 2)
    neck_r = int(r * 0.1)

    # Base
    for i in range(int(r*0.1), int(r*0.3)):
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= base_r:
                    v[i, j, k] = 1.0

    # Body
    for i in range(int(r*0.3), int(r*0.75)):
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= body_r:
                    v[i, j, k] = 1.0

    # Neck
    for i in range(int(r*0.75), int(r*0.95)):
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= neck_r:
                    v[i, j, k] = 1.0

    v = ndimage.gaussian_filter(v, sigma=1.5)
    return v > 0.3


def create_smooth_mug(resolution=48):
    """Create mug with handle"""
    v = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution
    center = r // 2

    outer_r = int(r * 0.25)
    inner_r = int(r * 0.20)

    # Hollow cylinder
    for i in range(int(r*0.2), int(r*0.7)):
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if inner_r <= dist <= outer_r:
                    v[i, j, k] = 1.0

    # Bottom
    for j in range(r):
        for k in range(r):
            dist = np.sqrt((j - center)**2 + (k - center)**2)
            if dist <= outer_r:
                for i in range(int(r*0.2), int(r*0.24)):
                    v[i, j, k] = 1.0

    # Handle
    handle_y = center + outer_r + 5
    for i in range(int(r*0.3), int(r*0.6)):
        for offset in range(-3, 4):
            hy = min(r-1, handle_y + abs(offset))
            hz = center + offset
            if 0 <= hy < r and 0 <= hz < r:
                v[i, hy, hz] = 1.0
                if hy + 1 < r:
                    v[i, hy+1, hz] = 1.0

    v = ndimage.gaussian_filter(v, sigma=1.0)
    return v > 0.3


def create_smooth_sofa(resolution=48):
    """Create sofa"""
    v = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    # Seat
    for i in range(int(r*0.35), int(r*0.5)):
        for j in range(int(r*0.2), int(r*0.8)):
            for k in range(int(r*0.35), min(int(r*0.75), r)):
                v[i, j, k] = 1.0

    # Back
    for i in range(int(r*0.5), min(int(r*0.85), r)):
        for j in range(int(r*0.2), int(r*0.35)):
            for k in range(int(r*0.35), min(int(r*0.75), r)):
                v[i, j, k] = 0.9

    # Arms
    for i in range(int(r*0.35), min(int(r*0.7), r)):
        for j in range(int(r*0.2), int(r*0.8)):
            for k_range in [(int(r*0.33), int(r*0.38)), (int(r*0.72), int(r*0.77))]:
                for k in range(k_range[0], min(k_range[1], r)):
                    v[i, j, k] = 0.85

    v = ndimage.gaussian_filter(v, sigma=1.5)
    return v > 0.2


def create_smooth_bed(resolution=48):
    """Create bed"""
    v = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    # Mattress
    for i in range(int(r*0.3), int(r*0.45)):
        for j in range(int(r*0.15), min(int(r*0.85), r)):
            for k in range(int(r*0.2), min(int(r*0.8), r)):
                v[i, j, k] = 1.0

    # Headboard
    for i in range(int(r*0.45), min(int(r*0.75), r)):
        for j in range(int(r*0.15), int(r*0.25)):
            for k in range(int(r*0.2), min(int(r*0.8), r)):
                v[i, j, k] = 0.9

    # 4 Legs
    leg_thick = 3
    for leg_j, leg_k in [(int(r*0.2), int(r*0.25)), (int(r*0.2), int(r*0.75)),
                          (int(r*0.8), int(r*0.25)), (int(r*0.8), int(r*0.75))]:
        for i in range(int(r*0.1), int(r*0.3)):
            for j in range(max(0, leg_j-leg_thick//2), min(r, leg_j+leg_thick//2)):
                for k in range(max(0, leg_k-leg_thick//2), min(r, leg_k+leg_thick//2)):
                    v[i, j, k] = 1.0

    v = ndimage.gaussian_filter(v, sigma=1.3)
    return v > 0.2
