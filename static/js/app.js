// VecViz - 3D Embedding Visualization
// Uses Three.js for WebGL rendering

let scene, camera, renderer, points;
let pointsData = [];
let raycaster, mouse;
let hoveredIndex = null;

// Initialize Three.js scene
function initScene() {
    const container = document.getElementById('scatter-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f0f23);

    // Camera
    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 3;

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Raycaster for mouse interaction
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.05;
    mouse = new THREE.Vector2();

    // Add orbit controls (manual implementation)
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };
    let rotationSpeed = 0.005;

    renderer.domElement.addEventListener('mousedown', (e) => {
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    renderer.domElement.addEventListener('mousemove', (e) => {
        // Update mouse for raycasting
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        if (isDragging) {
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            if (points) {
                points.rotation.y += deltaX * rotationSpeed;
                points.rotation.x += deltaY * rotationSpeed;
            }

            previousMousePosition = { x: e.clientX, y: e.clientY };
        }
    });

    renderer.domElement.addEventListener('mouseup', () => {
        isDragging = false;
    });

    renderer.domElement.addEventListener('mouseleave', () => {
        isDragging = false;
    });

    // Zoom with scroll
    renderer.domElement.addEventListener('wheel', (e) => {
        e.preventDefault();
        camera.position.z += e.deltaY * 0.002;
        camera.position.z = Math.max(1, Math.min(10, camera.position.z));
    });

    // Handle resize
    window.addEventListener('resize', () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    });

    // Animation loop
    animate();
}

function animate() {
    requestAnimationFrame(animate);

    // Check for hover
    if (points && pointsData.length > 0) {
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(points);

        if (intersects.length > 0) {
            const index = intersects[0].index;
            if (index !== hoveredIndex) {
                hoveredIndex = index;
                document.getElementById('point-text').textContent = pointsData[index].text;
            }
        }
    }

    renderer.render(scene, camera);
}

function updatePoints(data) {
    pointsData = data;

    // Remove existing points
    if (points) {
        scene.remove(points);
        points.geometry.dispose();
        points.material.dispose();
    }

    if (data.length === 0) {
        document.getElementById('point-count').textContent = '0 points';
        return;
    }

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(data.length * 3);
    const colors = new Float32Array(data.length * 3);

    // Color palette
    const colorPalette = [
        new THREE.Color(0x4a9eff),
        new THREE.Color(0x2ecc71),
        new THREE.Color(0xe74c3c),
        new THREE.Color(0xf39c12),
        new THREE.Color(0x9b59b6),
        new THREE.Color(0x1abc9c),
        new THREE.Color(0xe91e63),
        new THREE.Color(0x00bcd4),
    ];

    for (let i = 0; i < data.length; i++) {
        positions[i * 3] = data[i].x;
        positions[i * 3 + 1] = data[i].y;
        positions[i * 3 + 2] = data[i].z;

        const color = colorPalette[i % colorPalette.length];
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create material
    const material = new THREE.PointsMaterial({
        size: 0.08,
        vertexColors: true,
        sizeAttenuation: true,
    });

    // Create points
    points = new THREE.Points(geometry, material);
    scene.add(points);

    document.getElementById('point-count').textContent = `${data.length} points`;
}

// API functions
async function fetchPoints() {
    try {
        const response = await fetch('/points');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to fetch points:', error);
        return { points: [], needs_update: false };
    }
}

async function submitEmbed(prompt) {
    const response = await fetch('/embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
    });
    return response.json();
}

async function recomputeTSNE() {
    const response = await fetch('/tsne/compute', { method: 'POST' });
    return response.json();
}

async function refreshVisualization() {
    const data = await fetchPoints();
    updatePoints(data.points || []);

    if (data.needs_update) {
        setStatus('New embeddings added. Click "Recompute t-SNE" to update.', 'warning');
    }
}

function setStatus(message, type = '') {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = type;
}

// Event listeners
document.getElementById('embed-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('prompt-input');
    const prompt = input.value.trim();
    if (!prompt) return;

    setStatus('Adding embedding...');
    try {
        const result = await submitEmbed(prompt);
        input.value = '';
        setStatus(`Embedding added (${result.embedding_dim}D). Recompute t-SNE to visualize.`, 'success');
    } catch (error) {
        setStatus('Failed to add embedding: ' + error.message, 'error');
    }
});

document.getElementById('recompute-btn').addEventListener('click', async () => {
    setStatus('Computing t-SNE...');
    try {
        const result = await recomputeTSNE();
        setStatus(`t-SNE complete: ${result.points_processed} points in ${result.computation_time_ms}ms`, 'success');
        await refreshVisualization();
    } catch (error) {
        setStatus('t-SNE failed: ' + error.message, 'error');
    }
});

// Initialize
initScene();
refreshVisualization();
