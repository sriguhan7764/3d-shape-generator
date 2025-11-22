from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import torch
import numpy as np
import json
import io
import base64
from models.gan_3d import Generator
from utils.shape_utils import voxel_to_obj, voxel_to_stl
import os

app = Flask(__name__)
CORS(app)

# Initialize the generator model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim=200, output_dim=32).to(device)

# Load pre-trained weights if available
model_path = 'checkpoints/generator.pth'
if os.path.exists(model_path):
    generator.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded pre-trained model")
else:
    print("No pre-trained model found - generating random shapes")

generator.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/api/generate', methods=['POST'])
def generate_shape():
    """Generate 3D shape from natural language description"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        category = data.get('category', 'chair')
        num_shapes = data.get('num_shapes', 1)

        # If prompt provided, use ADVANCED NLP + TOPOLOGY OPTIMIZATION
        if prompt and len(prompt.strip()) > 0:
            print(f"📝 Processing NLP prompt: {prompt}")
            from utils.nlp_processor import DesignConstraintParser, generate_from_constraints
            from utils.topology_optimizer import generate_optimized_design, apply_material_properties
            from utils.marching_cubes import voxels_to_smooth_mesh

            parser = DesignConstraintParser()
            constraints = parser.parse(prompt)

            print(f"✅ Extracted constraints: {constraints}")

            # Check if user wants optimization
            wants_optimization = any(word in prompt.lower() for word in [
                'optimize', 'best', 'efficient', 'lightweight', 'strong', 'sturdy', 'alternatives', 'options'
            ])

            if wants_optimization:
                print(f"🔬 Running TOPOLOGY OPTIMIZATION...")

                # Generate multiple optimized alternatives
                alternatives = generate_optimized_design(constraints, num_alternatives=3)

                # Convert best alternative to mesh
                best = alternatives[0]
                vertices, faces = voxels_to_smooth_mesh(best['voxels'], level=0.3, smooth_iterations=4)

                # Get material properties
                material = 'plastic'  # Default
                if constraints.get('material'):
                    material = constraints['material'][0] if isinstance(constraints['material'], list) else constraints['material']

                material_props = apply_material_properties(best['voxels'], material)

                response = {
                    'success': True,
                    'vertices': vertices.tolist(),
                    'faces': faces.tolist(),
                    'message': f'✨ Optimized design from: "{prompt}"',
                    'constraints': constraints,
                    'optimized': True,
                    'metrics': best['metrics'],
                    'material_properties': material_props,
                    'alternatives_count': len(alternatives),
                    'rank': 1
                }

                return jsonify(response)
            else:
                # Standard parametric generation
                voxel_grid = generate_from_constraints(constraints)
                vertices, faces = voxels_to_smooth_mesh(voxel_grid, level=0.3, smooth_iterations=4)

                response = {
                    'success': True,
                    'vertices': vertices.tolist(),
                    'faces': faces.tolist(),
                    'voxel_shape': voxel_grid.shape,
                    'message': f'Generated from prompt: "{prompt}"',
                    'constraints': constraints
                }

                return jsonify(response)

        # PROFESSIONAL QUALITY: Use Marching Cubes for smooth meshes
        if not os.path.exists(model_path):
            from utils.marching_cubes import (
                create_parametric_chair, create_parametric_table,
                create_parametric_bottle, voxels_to_smooth_mesh
            )
            from utils.advanced_shapes import (
                create_smooth_mug, create_smooth_sofa, create_smooth_bed
            )

            # Create HIGH-RESOLUTION shapes (64x64x64)
            print(f"Generating PROFESSIONAL QUALITY {category}...")

            if category == 'chair':
                voxel_grid = create_parametric_chair(resolution=64)
            elif category == 'table':
                voxel_grid = create_parametric_table(resolution=64)
            elif category == 'bottle':
                voxel_grid = create_parametric_bottle(resolution=64)
            elif category == 'mug':
                voxel_grid = create_smooth_mug(resolution=64)
            elif category == 'sofa':
                voxel_grid = create_smooth_sofa(resolution=64)
            elif category == 'bed':
                voxel_grid = create_smooth_bed(resolution=64)
            else:
                # Sphere for others
                from utils.shape_utils import create_sample_shape
                voxel_grid = create_sample_shape('sphere', 64)

            # Use MARCHING CUBES for ultra-smooth mesh
            vertices, faces = voxels_to_smooth_mesh(voxel_grid, level=0.3, smooth_iterations=4)

        else:
            # Generate random latent vectors
            z = torch.randn(num_shapes, 200).to(device)

            # Generate shapes
            with torch.no_grad():
                generated = generator(z)
                voxels = generated.cpu().numpy()

            # Convert using marching cubes
            from utils.marching_cubes import voxels_to_smooth_mesh
            vertices, faces = voxels_to_smooth_mesh(voxels[0, 0], level=0.3)

        response = {
            'success': True,
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'voxel_shape': voxel_grid.shape,
            'message': f'Generated {num_shapes} shape(s) for category: {category}'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export/<format>', methods=['POST'])
def export_shape(format):
    """Export generated shape to various formats"""
    try:
        data = request.json
        vertices = np.array(data['vertices'])
        faces = np.array(data['faces'])

        if format == 'obj':
            obj_data = voxel_to_obj(vertices, faces)
            return jsonify({'success': True, 'data': obj_data, 'format': 'obj'})

        elif format == 'stl':
            stl_data = voxel_to_stl(vertices, faces)
            # Convert binary STL to base64 for transfer
            stl_b64 = base64.b64encode(stl_data).decode('utf-8')
            return jsonify({'success': True, 'data': stl_b64, 'format': 'stl'})

        else:
            return jsonify({'success': False, 'error': 'Unsupported format'}), 400

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/categories')
def get_categories():
    """Get available shape categories"""
    categories = [
        'chair', 'table', 'lamp', 'airplane', 'car',
        'sofa', 'bottle', 'mug', 'guitar', 'bed'
    ]
    return jsonify({'categories': categories})

@app.route('/api/model/info')
def model_info():
    """Get model information"""
    info = {
        'device': str(device),
        'model_loaded': os.path.exists(model_path),
        'latent_dim': 200,
        'output_resolution': 32,
        'supported_formats': ['obj', 'stl']
    }
    return jsonify(info)

def voxel_to_mesh(voxel_grid, threshold=0.3):
    """Convert voxel grid to mesh vertices and faces (optimized)"""
    from scipy import ndimage

    # Get the coordinates of filled voxels
    coords = np.argwhere(voxel_grid > threshold)

    if len(coords) == 0:
        # Return a simple shape if no voxels
        vertices = np.array([
            [0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0],
            [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5]
        ], dtype=float)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]
        ])
        return vertices, faces

    # Limit to reasonable number for performance
    if len(coords) > 5000:
        # Sample randomly to reduce complexity
        indices = np.random.choice(len(coords), 5000, replace=False)
        coords = coords[indices]

    # Create vertices for each voxel (simplified cube mesh)
    vertices = []
    faces = []

    for i, coord in enumerate(coords):
        x, y, z = coord
        # Add 8 vertices for the cube
        base_idx = len(vertices)
        vertices.extend([
            [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
            [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
        ])

        # Add 12 triangular faces for the cube
        faces.extend([
            [base_idx+0, base_idx+1, base_idx+2], [base_idx+0, base_idx+2, base_idx+3],
            [base_idx+4, base_idx+5, base_idx+6], [base_idx+4, base_idx+6, base_idx+7],
            [base_idx+0, base_idx+1, base_idx+5], [base_idx+0, base_idx+5, base_idx+4],
            [base_idx+2, base_idx+3, base_idx+7], [base_idx+2, base_idx+7, base_idx+6],
            [base_idx+0, base_idx+3, base_idx+7], [base_idx+0, base_idx+7, base_idx+4],
            [base_idx+1, base_idx+2, base_idx+6], [base_idx+1, base_idx+6, base_idx+5]
        ])

    return np.array(vertices, dtype=float), np.array(faces, dtype=int)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    print("=" * 60)
    print("🚀 3D Generative Design for Additive Manufacturing")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model loaded: {os.path.exists(model_path)}")
    print("=" * 60)
    print("Starting server on http://localhost:8080")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=8080)
