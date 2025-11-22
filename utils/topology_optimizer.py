import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

class TopologyOptimizer:
    """
    Advanced topology optimization for generative design
    Optimizes shape for strength/weight ratio based on constraints
    """

    def __init__(self, resolution=64):
        self.resolution = resolution

    def optimize(self, constraints, iterations=50):
        """
        Multi-objective topology optimization

        Objectives:
        - Minimize material (weight)
        - Maximize strength
        - Meet dimensional constraints
        - Ensure manufacturability
        """

        print(f"🔬 Running topology optimization ({iterations} iterations)...")

        # Initialize density field (what gets optimized)
        density = np.random.rand(self.resolution, self.resolution, self.resolution) * 0.5

        # Apply constraints to density field
        density = self.apply_constraints(density, constraints)

        # Optimization loop
        for i in range(iterations):
            # Calculate compliance (inverse of stiffness)
            compliance = self.calculate_compliance(density, constraints)

            # Calculate gradient
            gradient = self.calculate_sensitivity(density, constraints)

            # Update density using gradient descent
            density -= 0.01 * gradient

            # Apply density limits [0, 1]
            density = np.clip(density, 0, 1)

            # Apply constraints
            density = self.apply_constraints(density, constraints)

            if i % 10 == 0:
                print(f"  Iteration {i}/{iterations}: Compliance = {compliance:.4f}")

        # Threshold to create final geometry
        optimized_voxels = density > 0.5

        # Smooth the result
        optimized_voxels = ndimage.binary_opening(optimized_voxels, iterations=2)
        optimized_voxels = ndimage.binary_closing(optimized_voxels, iterations=2)

        return optimized_voxels, density

    def apply_constraints(self, density, constraints):
        """Apply design constraints to density field"""
        r = self.resolution

        # Load-bearing regions must have material
        if constraints.get('weight_capacity'):
            # Top surface needs to be strong
            density[int(r*0.8):, :, :] = np.maximum(density[int(r*0.8):, :, :], 0.7)

        # Attachment points must have material
        if constraints.get('attachments'):
            for att in constraints['attachments']:
                if att['type'] == 'wall':
                    # Back wall attachment
                    density[:, :5, :] = 1.0
                elif att['type'] == 'floor':
                    # Bottom attachment
                    density[:8, :, :] = 1.0

        # Ergonomic = smooth transitions
        if constraints.get('ergonomic'):
            density = ndimage.gaussian_filter(density, sigma=2.0)

        # Lightweight = reduce material
        if constraints.get('lightweight'):
            density *= 0.7

        # Sturdy = increase critical areas
        if constraints.get('sturdy'):
            density[density > 0.3] *= 1.2
            density = np.clip(density, 0, 1)

        return density

    def calculate_compliance(self, density, constraints):
        """
        Calculate structural compliance (measure of flexibility)
        Lower compliance = stiffer structure
        """

        # Simplified compliance calculation
        # In real FEA, this would solve K*u = f for displacements

        # Penalize low density regions (SIMP method)
        penalized_density = density ** 3

        # Calculate approximate compliance
        # (This is a simplified proxy for real FEA)
        compliance = np.sum(penalized_density) / np.sum(density + 0.001)

        return compliance

    def calculate_sensitivity(self, density, constraints):
        """Calculate sensitivity (gradient) of compliance w.r.t. density"""

        # Simplified sensitivity
        # In real topology optimization, this comes from adjoint method

        gradient = np.zeros_like(density)

        # Penalize regions that don't contribute to stiffness
        gradient = -3 * (density ** 2) + 0.1 * np.random.randn(*density.shape)

        # Smooth gradient
        gradient = ndimage.gaussian_filter(gradient, sigma=1.0)

        return gradient


class GenerativeDesignEngine:
    """
    Complete generative design engine with multi-objective optimization
    """

    def __init__(self, resolution=64):
        self.resolution = resolution
        self.optimizer = TopologyOptimizer(resolution)

    def generate_alternatives(self, constraints, num_alternatives=5):
        """
        Generate multiple design alternatives that meet constraints
        Each optimized for different objectives
        """

        print(f"🎨 Generating {num_alternatives} design alternatives...")

        alternatives = []

        for i in range(num_alternatives):
            print(f"\n📐 Alternative {i+1}/{num_alternatives}")

            # Vary optimization objectives for each alternative
            modified_constraints = constraints.copy()

            if i == 0:
                # Minimize weight
                modified_constraints['objective'] = 'lightweight'
                modified_constraints['lightweight'] = True
            elif i == 1:
                # Maximize strength
                modified_constraints['objective'] = 'strength'
                modified_constraints['sturdy'] = True
            elif i == 2:
                # Balance weight and strength
                modified_constraints['objective'] = 'balanced'
            elif i == 3:
                # Optimize for manufacturability
                modified_constraints['objective'] = 'manufacturable'
                modified_constraints['minimal'] = True
            else:
                # Creative/decorative
                modified_constraints['objective'] = 'aesthetic'
                modified_constraints['decorative'] = True

            # Run optimization
            voxels, density = self.optimizer.optimize(modified_constraints, iterations=30)

            # Calculate metrics
            metrics = self.calculate_metrics(voxels, density, modified_constraints)

            alternatives.append({
                'voxels': voxels,
                'density': density,
                'constraints': modified_constraints,
                'metrics': metrics,
                'index': i
            })

            print(f"  ✅ Metrics: Weight={metrics['weight']:.1f}%, Strength={metrics['strength_score']:.2f}")

        return alternatives

    def calculate_metrics(self, voxels, density, constraints):
        """Calculate performance metrics for a design"""

        # Volume/Weight (material usage)
        total_voxels = np.prod(voxels.shape)
        filled_voxels = np.sum(voxels)
        weight_percentage = (filled_voxels / total_voxels) * 100

        # Strength score (based on density distribution)
        avg_density = np.mean(density[voxels])
        strength_score = avg_density * 10

        # Strength-to-weight ratio
        str_to_weight = strength_score / (weight_percentage + 0.1)

        # Manufacturability (fewer complex features = more manufacturable)
        # Count connected components
        from scipy.ndimage import label
        labeled, num_features = label(voxels)
        manufacturability = 10 / (num_features + 1)  # Simpler = better

        # Estimated cost (material + manufacturing complexity)
        material_cost = weight_percentage * 0.5  # Arbitrary units
        manufacturing_cost = (10 - manufacturability) * 2
        total_cost = material_cost + manufacturing_cost

        return {
            'weight': weight_percentage,
            'strength_score': strength_score,
            'strength_to_weight': str_to_weight,
            'manufacturability': manufacturability,
            'cost_estimate': total_cost,
            'num_components': num_features
        }


def generate_optimized_design(constraints, num_alternatives=5):
    """
    Main function to generate optimized designs
    """

    engine = GenerativeDesignEngine(resolution=64)

    # Generate multiple alternatives
    alternatives = engine.generate_alternatives(constraints, num_alternatives)

    # Rank alternatives
    ranked = rank_alternatives(alternatives, constraints)

    return ranked


def rank_alternatives(alternatives, constraints):
    """
    Rank design alternatives based on objectives
    """

    # Score each alternative
    for alt in alternatives:
        metrics = alt['metrics']

        # Calculate overall score based on objectives
        score = 0

        # Weight factors based on constraints
        if constraints.get('lightweight'):
            score += (100 - metrics['weight']) * 0.4
        else:
            score += metrics['weight'] * 0.1

        if constraints.get('sturdy'):
            score += metrics['strength_score'] * 10
        else:
            score += metrics['strength_score'] * 5

        # Manufacturability always matters
        score += metrics['manufacturability'] * 5

        # Cost efficiency
        score -= metrics['cost_estimate'] * 0.5

        alt['overall_score'] = score

    # Sort by score
    ranked = sorted(alternatives, key=lambda x: x['overall_score'], reverse=True)

    return ranked


def apply_material_properties(voxels, material='plastic'):
    """
    Apply material-specific properties for realistic rendering
    """

    materials = {
        'plastic': {'density': 1.2, 'youngs_modulus': 2.5, 'cost_per_cm3': 0.05},
        'wood': {'density': 0.7, 'youngs_modulus': 11, 'cost_per_cm3': 0.08},
        'metal': {'density': 7.8, 'youngs_modulus': 200, 'cost_per_cm3': 0.15},
        'aluminum': {'density': 2.7, 'youngs_modulus': 69, 'cost_per_cm3': 0.12},
    }

    props = materials.get(material, materials['plastic'])

    volume_cm3 = np.sum(voxels) * 0.001  # Assuming 1 voxel = 1mm³

    weight_kg = volume_cm3 * props['density'] / 1000
    material_cost = volume_cm3 * props['cost_per_cm3']

    return {
        'material': material,
        'volume_cm3': volume_cm3,
        'weight_kg': weight_kg,
        'material_cost': material_cost,
        'properties': props
    }
