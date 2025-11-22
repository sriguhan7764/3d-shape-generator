# 3D Generative Design for Additive Manufacturing

An AI-powered system for generating novel 3D shapes suitable for 3D printing using Generative Adversarial Networks (GANs) trained on the ShapeNet dataset.

## Overview

This project implements a 3D GAN architecture that generates unique 3D models in voxel format, which can be exported as OBJ and STL files for 3D printing. The system features an interactive web interface with real-time 3D visualization, making it easy to generate, view, and export custom shapes.

## Features

- **3D GAN Architecture**: State-of-the-art 3D Generative Adversarial Network for voxel-based shape generation
- **Interactive Web Interface**: Beautiful web UI with real-time 3D visualization using Three.js
- **Multiple Export Formats**: Export generated shapes as OBJ and STL files for 3D printing
- **ShapeNet Integration**: Trained on the industry-standard ShapeNet dataset (55+ categories)
- **Real-time Statistics**: View mesh complexity, vertex/face counts, and printability info
- **Category-based Generation**: Generate shapes across multiple categories (chairs, tables, vehicles, etc.)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (8GB+ recommended for training)
- GPU with CUDA support (optional, for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sriguhan7764/3d-shape-generator.git
cd 3d-shape-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the web application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Web Interface

1. **Select Category**: Choose the type of shape you want to generate (chair, table, lamp, etc.)
2. **Generate Shape**: Click "Generate Shape" to create a new 3D model
3. **View in 3D**: Rotate, zoom, and inspect the generated shape in the interactive viewer
4. **Export**: Download the shape as OBJ or STL for 3D printing

### Training Your Own Model

#### Download Dataset

For training, you can use the sample synthetic dataset or download the full ShapeNet:

```bash
# Create sample synthetic dataset (quick start)
python download_shapenet.py --source sample --num_samples 100

# Or get instructions for downloading full ShapeNet
python download_shapenet.py --source instructions
```

#### Train the Model

```bash
# Train on synthetic data
python train.py --data_dir data/shapenet --num_epochs 100

# Train with custom settings
python train.py --data_dir data/shapenet --batch_size 32 --lr 0.0001 --num_epochs 200
```

Training on a GPU is highly recommended. Expected training times:
- ~1-2 hours for 100 epochs on synthetic data
- ~24-48 hours for 100 epochs on full ShapeNet

## Project Structure

```
3d-shape-generator/
├── app.py                  # Flask web application
├── train.py               # GAN training script
├── download_shapenet.py   # Dataset download utility
├── requirements.txt       # Python dependencies
├── start.sh              # Quick start script
├── .gitignore            # Git ignore rules
├── models/
│   ├── __init__.py
│   └── gan_3d.py         # 3D GAN architecture
├── utils/
│   ├── __init__.py
│   ├── shape_utils.py    # 3D shape utilities
│   ├── marching_cubes.py # Mesh generation
│   ├── topology_optimizer.py
│   ├── advanced_shapes.py
│   └── nlp_processor.py
├── templates/
│   └── index.html        # Web interface
├── static/
│   └── app.js            # Frontend JavaScript
├── checkpoints/          # Saved model weights (create this)
└── data/                 # Dataset storage (create this)
```

## Model Architecture

### Generator
- **Input**: 200-dimensional latent vector
- **Architecture**: 5-layer transposed 3D convolution network
- **Output**: 32×32×32 voxel grid
- **Activation**: Sigmoid (binary voxel occupancy)

### Discriminator
- **Input**: 32×32×32 voxel grid
- **Architecture**: 5-layer 3D convolution network
- **Output**: Real/fake classification
- **Activation**: LeakyReLU + Sigmoid

The model uses the WGAN-GP (Wasserstein GAN with Gradient Penalty) approach for stable training.

## API Documentation

### Endpoints

- `GET /` - Web interface
- `POST /api/generate` - Generate new shape
  - Body: `{"category": "chair"}`
  - Response: `{"voxels": [...], "mesh": {...}}`
- `POST /api/export/<format>` - Export shape (obj/stl)
  - Body: `{"voxels": [...]}`
  - Response: File download
- `GET /api/categories` - Get available categories
- `GET /api/model/info` - Get model information

## Dataset

The project uses the **ShapeNet** dataset:

- **Size**: 50,000+ unique 3D models
- **Categories**: 55 common object categories
- **Format**: Voxel grids (32³ resolution)
- **Sources**:
  - Official: [shapenet.org](https://shapenet.org/)
  - Kaggle: [ShapeNet Dataset](https://www.kaggle.com/datasets/balraj98/shapenet-dataset)
  - HuggingFace: [ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)

## Export Formats

### OBJ Format
- Text-based format
- Widely supported by 3D software
- Best for editing and visualization

### STL Format
- Binary format
- Industry standard for 3D printing
- Direct slicing for print preparation

## Customization

### Add New Categories

Edit `app.py` to add categories:

```python
categories = [
    'chair', 'table', 'lamp', 'airplane', 'car',
    'your_new_category'  # Add here
]
```

### Adjust Model Resolution

Change resolution in `models/gan_3d.py`:

```python
generator = Generator(latent_dim=200, output_dim=64)  # 64x64x64
```

**Note**: Higher resolution requires more memory and training time.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model generates random shapes | Train the model first or use pre-trained weights |
| Out of memory during training | Reduce batch size or model resolution |
| Poor shape quality | Train for more epochs or increase model capacity |
| Import errors | Make sure all dependencies are installed: `pip install -r requirements.txt` |

## Roadmap

- [ ] Diffusion model implementation (state-of-the-art)
- [ ] Conditional generation (specify constraints)
- [ ] Point cloud support
- [ ] Multi-resolution generation
- [ ] Style transfer between shapes
- [ ] Topology optimization
- [ ] Support constraints (e.g., "must attach here")
- [ ] Pre-trained model weights
- [ ] Docker support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{shapenet_generative_design,
  title={3D Generative Design for Additive Manufacturing},
  author={Guhan S},
  year={2025},
  url={https://github.com/sriguhan7764/3d-shape-generator}
}
```

## Acknowledgments

- [ShapeNet Dataset](https://shapenet.org/) - For providing the 3D model dataset
- [3D-GAN Paper](https://arxiv.org/abs/1610.07584) - Original 3D-GAN research
- [Three.js](https://threejs.org/) - 3D visualization library

## Support

For issues and questions:
- Open a [GitHub Issue](https://github.com/sriguhan7764/3d-shape-generator/issues)
- Contact: guhan_s@zohomail.in

## Star History

If you find this project useful, please consider giving it a star!

---

Built with passion for the generative AI and additive manufacturing community
