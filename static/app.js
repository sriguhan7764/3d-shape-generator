// Global variables
let scene, camera, renderer, controls, currentMesh;
let currentVertices = null;
let currentFaces = null;

// Initialize Three.js scene
function initViewer() {
    const container = document.getElementById('viewer');

    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f7fa);

    // Create camera
    camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(40, 40, 40);
    camera.lookAt(0, 0, 0);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(50, 50, 50);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-50, -50, -50);
    scene.add(directionalLight2);

    // Add grid helper
    const gridHelper = new THREE.GridHelper(50, 50, 0x888888, 0xcccccc);
    scene.add(gridHelper);

    // Add axes helper
    const axesHelper = new THREE.AxesHelper(25);
    scene.add(axesHelper);

    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 10;
    controls.maxDistance = 200;

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Add welcome text
    showWelcomeMessage();

    // Animation loop
    animate();

    // Load model info
    loadModelInfo();
}

function onWindowResize() {
    const container = document.getElementById('viewer');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function showWelcomeMessage() {
    const loader = new THREE.FontLoader();
    // For now, just add a simple text mesh or placeholder
    console.log('3D Viewer initialized. Click "Generate Shape" to begin.');
}

// Generate new shape
async function generateShape() {
    const prompt = document.getElementById('prompt').value.trim();
    const category = document.getElementById('category').value;

    // Show loading
    document.getElementById('loading').classList.add('active');
    document.getElementById('generate-btn').disabled = true;
    hideMessage();

    // Build request based on whether prompt or category is used
    const requestData = {};
    if (prompt && prompt.length > 0) {
        requestData.prompt = prompt;
    } else {
        requestData.category = category;
        requestData.num_shapes = 1;
    }

    console.log('🚀 Generating with:', requestData);

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (data.success) {
            console.log('✅ SHAPE RECEIVED:', data.vertices.length, 'vertices,', data.faces.length, 'faces');

            if (data.constraints) {
                console.log('📋 Constraints understood:', data.constraints);
            }

            currentVertices = data.vertices;
            currentFaces = data.faces;

            displayMesh(data.vertices, data.faces);
            updateStats(data.vertices, data.faces);

            // Show message with constraints and optimization metrics
            let message = data.message + ' (' + data.vertices.length + ' vertices)';

            if (data.optimized && data.metrics) {
                const m = data.metrics;
                message += '\n\n🏆 OPTIMIZATION METRICS:';
                message += '\n━━━━━━━━━━━━━━━━━━━━━━';
                message += `\n⚖️  Material Usage: ${m.weight.toFixed(1)}%`;
                message += `\n💪 Strength Score: ${m.strength_score.toFixed(2)}/10`;
                message += `\n📊 Strength/Weight: ${m.strength_to_weight.toFixed(2)}`;
                message += `\n🏭 Manufacturability: ${m.manufacturability.toFixed(2)}/10`;
                message += `\n💰 Est. Cost: $${m.cost_estimate.toFixed(2)}`;

                if (data.material_properties) {
                    const mp = data.material_properties;
                    message += '\n\n📦 MATERIAL: ' + mp.material.toUpperCase();
                    message += `\n   Weight: ${mp.weight_kg.toFixed(3)} kg`;
                    message += `\n   Volume: ${mp.volume_cm3.toFixed(1)} cm³`;
                    message += `\n   Material Cost: $${mp.material_cost.toFixed(2)}`;
                }

                if (data.alternatives_count > 1) {
                    message += `\n\n✨ Generated ${data.alternatives_count} alternatives (showing best)`;
                }
            } else if (data.constraints) {
                const c = data.constraints;
                let details = [];
                if (c.object_type && c.object_type !== 'custom') details.push(`Type: ${c.object_type}`);
                if (c.ergonomic) details.push('Ergonomic');
                if (c.sturdy) details.push('Sturdy');
                if (c.lightweight) details.push('Lightweight');
                if (c.features && c.features.length > 0) details.push(`Features: ${c.features.join(', ')}`);
                if (details.length > 0) {
                    message += '\n✅ ' + details.join(' | ');
                }
            }

            showMessage(message, 'success');

            // Enable export buttons
            document.getElementById('export-obj-btn').disabled = false;
            document.getElementById('export-stl-btn').disabled = false;
        } else {
            showMessage('Error: ' + data.error, 'error');
        }
    } catch (error) {
        showMessage('Failed to generate shape: ' + error.message, 'error');
    } finally {
        document.getElementById('loading').classList.remove('active');
        document.getElementById('generate-btn').disabled = false;
    }
}

// Display mesh in viewer
function displayMesh(vertices, faces) {
    // Remove existing mesh
    if (currentMesh) {
        scene.remove(currentMesh);
        currentMesh.geometry.dispose();
        currentMesh.material.dispose();
    }

    // Create geometry
    const geometry = new THREE.BufferGeometry();

    // Convert vertices and faces to Float32Array
    const verticesArray = new Float32Array(vertices.flat());
    const indicesArray = new Uint32Array(faces.flat());

    geometry.setAttribute('position', new THREE.BufferAttribute(verticesArray, 3));
    geometry.setIndex(new THREE.BufferAttribute(indicesArray, 1));
    geometry.computeVertexNormals();

    // Center the geometry
    geometry.center();

    // Create material with both solid and wireframe
    const material = new THREE.MeshPhongMaterial({
        color: 0x667eea,
        shininess: 100,
        specular: 0x444444,
        flatShading: false
    });

    // Create mesh
    currentMesh = new THREE.Mesh(geometry, material);
    currentMesh.castShadow = true;
    currentMesh.receiveShadow = true;

    scene.add(currentMesh);

    // Add wireframe overlay
    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeMaterial = new THREE.LineBasicMaterial({
        color: 0x000000,
        opacity: 0.1,
        transparent: true
    });
    const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
    currentMesh.add(wireframe);

    // Reset camera position
    const boundingBox = new THREE.Box3().setFromObject(currentMesh);
    const center = boundingBox.getCenter(new THREE.Vector3());
    const size = boundingBox.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 2.5; // Offset multiplier

    camera.position.set(cameraZ, cameraZ, cameraZ);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
}

// Update statistics
function updateStats(vertices, faces) {
    document.getElementById('stat-vertices').textContent = vertices.length.toLocaleString();
    document.getElementById('stat-faces').textContent = faces.length.toLocaleString();

    // Calculate complexity (simple heuristic based on face count)
    const complexity = Math.min(100, Math.round((faces.length / 10000) * 100));
    document.getElementById('stat-complexity').textContent = complexity + '%';

    // Assume printable for now
    document.getElementById('stat-printable').textContent = 'Yes';
}

// Export shape
async function exportShape(format) {
    if (!currentVertices || !currentFaces) {
        showMessage('No shape to export. Generate a shape first.', 'error');
        return;
    }

    try {
        const response = await fetch(`/api/export/${format}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                vertices: currentVertices,
                faces: currentFaces
            })
        });

        const data = await response.json();

        if (data.success) {
            // Create download
            let blob;
            let filename;

            if (format === 'obj') {
                blob = new Blob([data.data], { type: 'text/plain' });
                filename = `generated_shape_${Date.now()}.obj`;
            } else if (format === 'stl') {
                // Decode base64 STL data
                const binaryString = atob(data.data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                blob = new Blob([bytes], { type: 'application/octet-stream' });
                filename = `generated_shape_${Date.now()}.stl`;
            }

            // Trigger download
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            showMessage(`Exported as ${format.toUpperCase()} successfully!`, 'success');
        } else {
            showMessage('Export failed: ' + data.error, 'error');
        }
    } catch (error) {
        showMessage('Export failed: ' + error.message, 'error');
    }
}

// Load model info
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model/info');
        const data = await response.json();

        const infoHtml = `
            <strong>Device:</strong> ${data.device}<br>
            <strong>Model Status:</strong> ${data.model_loaded ? '✅ Loaded' : '⚠️ Random Generation'}<br>
            <strong>Latent Dim:</strong> ${data.latent_dim}<br>
            <strong>Resolution:</strong> ${data.output_resolution}³ voxels<br>
            <strong>Formats:</strong> ${data.supported_formats.join(', ')}
        `;

        document.getElementById('model-info').innerHTML = infoHtml;
    } catch (error) {
        document.getElementById('model-info').textContent = 'Failed to load model info';
    }
}

// Show message
function showMessage(text, type) {
    const messageDiv = document.getElementById('message');
    messageDiv.textContent = text;
    messageDiv.className = `message ${type} active`;

    setTimeout(() => {
        hideMessage();
    }, 5000);
}

// Hide message
function hideMessage() {
    const messageDiv = document.getElementById('message');
    messageDiv.classList.remove('active');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initViewer();
});
