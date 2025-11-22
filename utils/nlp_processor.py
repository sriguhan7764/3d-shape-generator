import re
import numpy as np

class DesignConstraintParser:
    """Parse natural language design requirements"""

    def __init__(self):
        self.object_types = {
            'chair': ['chair', 'seat', 'sitting'],
            'table': ['table', 'desk', 'workstation'],
            'bottle': ['bottle', 'container', 'flask'],
            'mug': ['mug', 'cup', 'glass'],
            'shelf': ['shelf', 'rack', 'storage'],
            'lamp': ['lamp', 'light', 'lighting'],
            'stand': ['stand', 'holder', 'support']
        }

    def parse(self, text):
        """Parse natural language into design constraints"""
        text = text.lower()

        constraints = {
            'object_type': self.extract_object_type(text),
            'dimensions': self.extract_dimensions(text),
            'weight_capacity': self.extract_weight(text),
            'height': self.extract_height(text),
            'width': self.extract_width(text),
            'depth': self.extract_depth(text),
            'material': self.extract_material(text),
            'style': self.extract_style(text),
            'features': self.extract_features(text),
            'attachments': self.extract_attachments(text),
            'ergonomic': 'ergonomic' in text or 'comfortable' in text,
            'lightweight': 'lightweight' in text or 'light' in text,
            'sturdy': 'sturdy' in text or 'strong' in text or 'robust' in text,
            'minimal': 'minimal' in text or 'simple' in text or 'clean' in text,
            'decorative': 'decorative' in text or 'ornate' in text or 'detailed' in text,
        }

        return constraints

    def extract_object_type(self, text):
        """Identify the object type"""
        for obj_type, keywords in self.object_types.items():
            if any(kw in text for kw in keywords):
                return obj_type
        return 'custom'

    def extract_dimensions(self, text):
        """Extract dimension specifications"""
        dims = {}

        # Look for patterns like "50cm tall", "30 inches wide"
        patterns = [
            r'(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in|ft|feet)\s*(tall|high|height)',
            r'(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in|ft|feet)\s*(wide|width)',
            r'(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in|ft|feet)\s*(deep|depth)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                dimension = match.group(3)
                dims[dimension] = {'value': value, 'unit': unit}

        return dims

    def extract_weight(self, text):
        """Extract weight capacity"""
        patterns = [
            r'support\s*(\d+\.?\d*)\s*(kg|lb|pounds|kilograms)',
            r'hold\s*(\d+\.?\d*)\s*(kg|lb|pounds|kilograms)',
            r'(\d+\.?\d*)\s*(kg|lb|pounds|kilograms)\s*capacity',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {'value': float(match.group(1)), 'unit': match.group(2)}
        return None

    def extract_height(self, text):
        """Extract height specification"""
        patterns = [
            r'(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in)\s*tall',
            r'height\s*of\s*(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {'value': float(match.group(1)), 'unit': match.group(2)}
        return None

    def extract_width(self, text):
        """Extract width specification"""
        patterns = [
            r'(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in)\s*wide',
            r'width\s*of\s*(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {'value': float(match.group(1)), 'unit': match.group(2)}
        return None

    def extract_depth(self, text):
        """Extract depth specification"""
        patterns = [
            r'(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in)\s*deep',
            r'depth\s*of\s*(\d+\.?\d*)\s*(cm|mm|m|inch|inches|in)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return {'value': float(match.group(1)), 'unit': match.group(2)}
        return None

    def extract_material(self, text):
        """Extract material preferences"""
        materials = []
        if 'wood' in text or 'wooden' in text:
            materials.append('wood')
        if 'metal' in text or 'steel' in text or 'aluminum' in text:
            materials.append('metal')
        if 'plastic' in text:
            materials.append('plastic')
        return materials if materials else None

    def extract_style(self, text):
        """Extract style preferences"""
        if 'modern' in text or 'contemporary' in text:
            return 'modern'
        if 'vintage' in text or 'classic' in text or 'traditional' in text:
            return 'classic'
        if 'industrial' in text:
            return 'industrial'
        if 'scandinavian' in text or 'nordic' in text:
            return 'scandinavian'
        return None

    def extract_features(self, text):
        """Extract specific features"""
        features = []
        if 'armrest' in text or 'arm rest' in text:
            features.append('armrests')
        if 'cushion' in text or 'padded' in text:
            features.append('cushioned')
        if 'wheels' in text or 'casters' in text:
            features.append('wheeled')
        if 'adjustable' in text:
            features.append('adjustable')
        if 'foldable' in text or 'folding' in text:
            features.append('foldable')
        if 'storage' in text or 'drawer' in text:
            features.append('storage')
        return features if features else None

    def extract_attachments(self, text):
        """Extract attachment points"""
        attachments = []

        # Look for "attach at", "must connect", etc.
        if 'attach' in text or 'mount' in text or 'connect' in text:
            if 'wall' in text:
                attachments.append({'type': 'wall', 'location': 'wall'})
            if 'floor' in text:
                attachments.append({'type': 'floor', 'location': 'floor'})
            if 'ceiling' in text:
                attachments.append({'type': 'ceiling', 'location': 'ceiling'})

        return attachments if attachments else None


def generate_from_constraints(constraints):
    """Generate parametric shape based on parsed constraints"""

    # Normalize dimensions to voxel space (64^3)
    resolution = 64

    # Base proportions
    if constraints['object_type'] == 'chair':
        return generate_constrained_chair(constraints, resolution)
    elif constraints['object_type'] == 'table':
        return generate_constrained_table(constraints, resolution)
    elif constraints['object_type'] == 'bottle':
        return generate_constrained_bottle(constraints, resolution)
    elif constraints['object_type'] == 'shelf':
        return generate_constrained_shelf(constraints, resolution)
    elif constraints['object_type'] == 'stand':
        return generate_constrained_stand(constraints, resolution)
    else:
        return generate_custom_object(constraints, resolution)


def generate_constrained_chair(constraints, resolution=64):
    """Generate chair based on constraints"""
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    # Extract dimensions or use defaults
    if constraints.get('height'):
        seat_height = int(r * 0.4)  # Adjust based on specified height
    else:
        seat_height = int(r * 0.5)

    width_factor = 0.4 if constraints.get('width') else 0.35

    # Sturdy = thicker legs
    leg_thickness = 5 if constraints.get('sturdy') else 3

    # Ergonomic = curved back
    back_curve = 1.5 if constraints.get('ergonomic') else 1.0

    # Seat
    seat_start_j = int(r * 0.25)
    seat_start_k = int(r * 0.25)
    seat_depth = int(r * 0.35)
    seat_width = int(r * width_factor)

    for i in range(seat_height, seat_height + 5):
        for j in range(seat_start_j, seat_start_j + seat_depth):
            for k in range(seat_start_k, seat_start_k + seat_width):
                # Cushioned = softer appearance
                if constraints.get('features') and 'cushioned' in constraints['features']:
                    center_j = seat_start_j + seat_depth // 2
                    center_k = seat_start_k + seat_width // 2
                    dist = np.sqrt((j - center_j)**2 + (k - center_k)**2)
                    voxels[i, j, k] = max(0, 1.0 - dist * 0.02)
                else:
                    voxels[i, j, k] = 1.0

    # Backrest
    back_height = int(r * 0.5) if constraints.get('ergonomic') else int(r * 0.4)
    for i in range(seat_height + 5, min(seat_height + 5 + back_height, r)):
        for j in range(seat_start_j, seat_start_j + 6):
            for k in range(seat_start_k, seat_start_k + seat_width):
                # Ergonomic curve
                height_progress = (i - seat_height - 5) / back_height
                curve_factor = 1.0 - 0.1 * height_progress * back_curve
                voxels[i, j, k] = curve_factor

    # Armrests if requested
    if constraints.get('features') and 'armrests' in constraints['features']:
        for i in range(seat_height, min(seat_height + 15, r)):
            # Left armrest
            for j in range(seat_start_j, seat_start_j + seat_depth):
                for k in range(seat_start_k - 4, seat_start_k):
                    voxels[i, j, k] = 0.9
            # Right armrest
            for j in range(seat_start_j, seat_start_j + seat_depth):
                for k in range(seat_start_k + seat_width, seat_start_k + seat_width + 4):
                    if k < r:
                        voxels[i, j, k] = 0.9

    # Legs
    leg_positions = [
        (seat_start_j + 3, seat_start_k + 3),
        (seat_start_j + 3, seat_start_k + seat_width - 3),
        (seat_start_j + seat_depth - 3, seat_start_k + 3),
        (seat_start_j + seat_depth - 3, seat_start_k + seat_width - 3)
    ]

    for leg_j, leg_k in leg_positions:
        for i in range(5, seat_height):
            for j in range(max(0, leg_j - leg_thickness), min(r, leg_j + leg_thickness)):
                for k in range(max(0, leg_k - leg_thickness), min(r, leg_k + leg_thickness)):
                    voxels[i, j, k] = 1.0

    return voxels


def generate_constrained_table(constraints, resolution=64):
    """Generate table based on constraints"""
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    # Height based on constraints
    table_height = int(r * 0.65)
    if constraints.get('height'):
        # Adjust based on specified height
        table_height = int(r * 0.7)

    # Weight capacity = thicker top
    top_thickness = 6 if constraints.get('weight_capacity') else 4

    # Width/depth
    width_factor = 0.8 if constraints.get('width') else 0.7

    # Table top
    for i in range(table_height, min(table_height + top_thickness, r)):
        for j in range(int(r*0.15), min(int(r*0.15 + r*width_factor), r)):
            for k in range(int(r*0.15), min(int(r*0.15 + r*width_factor), r)):
                voxels[i, j, k] = 1.0

    # Legs (sturdy = thicker)
    leg_thickness = 6 if constraints.get('sturdy') else 4

    leg_positions = [
        (int(r*0.2), int(r*0.2)),
        (int(r*0.2), min(int(r*0.8), r-1)),
        (min(int(r*0.8), r-1), int(r*0.2)),
        (min(int(r*0.8), r-1), min(int(r*0.8), r-1))
    ]

    for leg_j, leg_k in leg_positions:
        for i in range(5, table_height):
            for j in range(max(0, leg_j - leg_thickness), min(r, leg_j + leg_thickness)):
                for k in range(max(0, leg_k - leg_thickness), min(r, leg_k + leg_thickness)):
                    voxels[i, j, k] = 1.0

    return voxels


def generate_constrained_bottle(constraints, resolution=64):
    """Generate bottle based on constraints"""
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution
    center = r // 2

    # Height affects proportions
    height_factor = 1.0
    if constraints.get('height'):
        height_factor = 1.2

    # Base
    for i in range(5, int(15 * height_factor)):
        radius = 12
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= radius:
                    voxels[i, j, k] = 1.0

    # Body
    for i in range(int(15 * height_factor), int(50 * height_factor)):
        radius = 9
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= radius:
                    voxels[i, j, k] = 1.0

    # Neck
    for i in range(int(50 * height_factor), int(60 * height_factor)):
        radius = 4
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= radius:
                    voxels[i, j, k] = 1.0

    return voxels


def generate_constrained_shelf(constraints, resolution=64):
    """Generate shelf based on constraints"""
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution

    # Number of shelves
    num_shelves = 3
    shelf_thickness = 3

    # Wall-mounted vs floor standing
    is_wall_mounted = False
    if constraints.get('attachments'):
        for att in constraints['attachments']:
            if att['type'] == 'wall':
                is_wall_mounted = True

    if is_wall_mounted:
        # Wall-mounted shelf
        for shelf_num in range(num_shelves):
            shelf_height = int(r * 0.3) + shelf_num * int(r * 0.25)
            for i in range(shelf_height, shelf_height + shelf_thickness):
                for j in range(int(r*0.1), int(r*0.4)):
                    for k in range(int(r*0.2), int(r*0.8)):
                        voxels[i, j, k] = 1.0
    else:
        # Floor standing shelf with sides
        for shelf_num in range(num_shelves):
            shelf_height = int(r * 0.2) + shelf_num * int(r * 0.25)
            for i in range(shelf_height, shelf_height + shelf_thickness):
                for j in range(int(r*0.2), int(r*0.8)):
                    for k in range(int(r*0.2), int(r*0.8)):
                        voxels[i, j, k] = 1.0

        # Side panels
        for i in range(int(r*0.2), min(int(r*0.9), r)):
            for j in range(int(r*0.2), int(r*0.8)):
                for k in range(int(r*0.2), int(r*0.23)):
                    voxels[i, j, k] = 0.9
                for k in range(int(r*0.77), int(r*0.8)):
                    if k < r:
                        voxels[i, j, k] = 0.9

    return voxels


def generate_constrained_stand(constraints, resolution=64):
    """Generate stand/holder"""
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution
    center = r // 2

    # Base
    for i in range(5, 10):
        for j in range(center - 15, center + 15):
            for k in range(center - 15, center + 15):
                if 0 <= j < r and 0 <= k < r:
                    voxels[i, j, k] = 1.0

    # Column
    for i in range(10, int(r * 0.7)):
        for j in range(center - 4, center + 4):
            for k in range(center - 4, center + 4):
                if 0 <= j < r and 0 <= k < r:
                    voxels[i, j, k] = 1.0

    # Top holder
    for i in range(int(r * 0.7), int(r * 0.75)):
        for j in range(center - 8, center + 8):
            for k in range(center - 8, center + 8):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if 4 <= dist <= 8 and 0 <= j < r and 0 <= k < r:
                    voxels[i, j, k] = 1.0

    return voxels


def generate_custom_object(constraints, resolution=64):
    """Generate custom object based on constraints"""
    # Default to a simple parametric form
    voxels = np.zeros((resolution, resolution, resolution), dtype=float)
    r = resolution
    center = r // 2

    for i in range(int(r * 0.2), int(r * 0.8)):
        for j in range(r):
            for k in range(r):
                dist = np.sqrt((j - center)**2 + (k - center)**2)
                if dist <= r * 0.3:
                    voxels[i, j, k] = 1.0

    return voxels
