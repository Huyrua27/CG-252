# 🚀 Computer Graphics Starter Code - Modern OpenGL with Python

A comprehensive starter code package for teaching **Modern OpenGL** (OpenGL 3.3+) using Python and PyOpenGL. This project provides reusable utility libraries and progressive examples demonstrating shader-based rendering, lighting models, and texture mapping.

---

## 📁 Project Structure

```
Sample/
│
├── libs/                          # Core utility libraries
│   ├── buffer.py                  # VAO, VBO, EBO management
│   ├── shader.py                  # Shader compilation & linking
│   ├── transform.py               # 3D transformation matrices & Trackball camera
│   └── camera.py                  # Camera utilities (extends Trackball)
│
├── triangle/                      # Example 1: Basic Triangle
│   ├── viewer.py                  # Main entry point
│   ├── triangle.py                # Triangle geometry classes
│   ├── flat.vert/frag             # Flat shading (constant color)
│   ├── color_interp.vert/frag     # Vertex color interpolation
│   ├── phong.vert/frag            # Phong shading (per-fragment lighting)
│   └── phongex.vert/frag          # Extended Phong with multiple materials
│
├── cube/                          # Example 2: 3D Cube
│   ├── viewer.py                  # Viewer with Trackball camera control
│   ├── cube.py                    # Cube geometry class
│   └── color_interp.vert/frag     # Vertex color interpolation
│
├── prism/                         # Example 3: Prism
│   ├── viewer.py                  # Viewer with Trackball camera control
│   ├── prism.py                   # Prism geometry class
│   └── color_interp.vert/frag     # Vertex color interpolation
│
├── patch/                         # Example 4: Surface Patch
│   ├── viewer.py                  # Main entry point
│   ├── patch.py                   # Patch geometry classes
│   ├── flat.vert/frag             # Flat shading
│   ├── color_interp.vert/frag     # Vertex color interpolation
│   ├── phong.vert/frag            # Phong shading
│   ├── phongex.vert/frag          # Extended Phong
│   └── textured/                  # Textured patch example
│       ├── TexturedPatch.py       # Textured patch class
│       ├── phong_texture.vert/frag # Phong + Texture mapping
│       └── image/                 # Texture images
│
├── texcube/                       # Example 5: Textured Cube
│   ├── viewer.py                  # Viewer with Trackball camera control
│   ├── texcube.py                 # Textured cube class
│   ├── color_interp.vert/frag     # Vertex color interpolation
│   ├── phong_texture.vert/frag    # Phong + Texture mapping
│   └── image/                     # Texture images
│
├── demo/                          # Demo scripts
│   └── transformation.py          # Matrix transformation examples
│
└── README.md                      # This file
```

---

## 🛠️ Requirements

### Python Version
- Python **≥ 3.8** (tested with Python 3.10+)

### Required Packages

```bash
pip install PyOpenGL PyOpenGL_accelerate opencv-python glfw numpy
```

### System Requirements
- OpenGL 3.3+ support
- Display server (not headless mode)
- GPU drivers with OpenGL support

### Tested Environment
- Python 3.10
- GLFW 3.3
- PyOpenGL 3.1.6+
- OpenCV 4.x
- NumPy 1.20+

---

## 🧪 How to Run Examples

Each example folder contains a `viewer.py` file. Navigate to the example directory and run:

### 🔺 Example 1: Triangle

```bash
cd triangle
python viewer.py
```

**Features:**
- Multiple shading models: flat, color interpolation, Phong, Extended Phong
- Demonstrates vertex attributes (position, color, normal)
- Shows interleaved vs separate buffer layouts

### 🧊 Example 2: Cube

```bash
cd cube
python viewer.py
```

**Features:**
- 3D cube with vertex colors
- Trackball camera control (drag to rotate, scroll to zoom, right-click drag to pan)
- Depth testing enabled
- Press `W` to toggle wireframe/point/fill modes

### 📐 Example 3: Prism

```bash
cd prism
python viewer.py
```

**Features:**
- Prism geometry demonstration
- Trackball camera control
- Element buffer (EBO) usage

### 🎨 Example 4: Surface Patch

```bash
cd patch
python viewer.py
```

**Features:**
- Multiple patches with different shaders
- Textured patch example (switch textures with keys 1/2)
- Demonstrates multiple materials per object

### 🎯 Example 5: Textured Cube

```bash
cd texcube
python viewer.py
```

**Features:**
- Cube with texture mapping
- Phong lighting combined with textures
- Trackball camera control

### 🔧 Demo: Transformations

```bash
cd demo
python transformation.py
```

**Features:**
- Matrix transformation examples
- Translate, scale, rotate demonstrations

---

## 🎨 Shader Types

This project demonstrates several shading techniques:

| Shader Type | Description | Location |
|------------|-------------|----------|
| **Flat** | Constant color per primitive | `triangle/flat.*`, `patch/flat.*` |
| **Color Interpolation** | Interpolate vertex colors | `*/color_interp.*` |
| **Phong** | Per-fragment Phong lighting | `triangle/phong.*`, `patch/phong.*` |
| **Extended Phong** | Phong with multiple materials | `triangle/phongex.*`, `patch/phongex.*` |
| **Phong + Texture** | Phong lighting with texture mapping | `*/phong_texture.*` |

### Shader Pipeline

1. **Vertex Shader** (`*.vert`):
   - Transforms vertex positions (Model-View-Projection)
   - Computes normal transformations
   - Passes data to fragment shader

2. **Fragment Shader** (`*.frag`):
   - Per-fragment lighting calculations
   - Texture sampling
   - Final color computation

---

## 🔍 Core Components

### `libs/buffer.py`

**VAO (Vertex Array Object)**
- Manages vertex attribute state
- Methods: `add_vbo()`, `add_ebo()`, `activate()`, `deactivate()`
- Supports interleaved and separate buffer layouts

**UManager (Uniform Manager)**
- Uploads uniforms to shader programs
- Methods: `upload_uniform_matrix4fv()`, `upload_uniform_vector3fv()`, etc.
- Texture management: `setup_texture()`

### `libs/shader.py`

**Shader Class**
- Compiles vertex and fragment shaders
- Links shader program
- Automatic resource cleanup
- Supports file paths or source strings

### `libs/transform.py`

**Matrix Functions**
- `translate()`, `rotate()`, `scale()` - Model transformations
- `perspective()`, `ortho()`, `frustum()` - Projection matrices
- `lookat()` - View matrix
- Quaternion utilities for rotations

**Trackball Class**
- Virtual trackball camera control
- Methods: `drag()`, `zoom()`, `pan()`
- Automatic view and projection matrix generation

### `libs/camera.py`

**Camera Class**
- Extends Trackball
- `place()` method for camera positioning

---

## 🧠 Learning Objectives

### Beginner Level
- ✅ Understand Modern OpenGL pipeline (no deprecated functions)
- ✅ Learn VAO/VBO/EBO concepts and usage
- ✅ Write basic GLSL shaders (vertex + fragment)
- ✅ Apply 3D transformations using matrices

### Intermediate Level
- ✅ Implement different shading models (flat, Gouraud-like, Phong)
- ✅ Understand per-vertex vs per-fragment lighting
- ✅ Work with vertex attributes and interleaved buffers
- ✅ Use element buffers for indexed rendering

### Advanced Level
- ✅ Combine lighting with texture mapping
- ✅ Implement multiple materials per object
- ✅ Camera control with Trackball interface
- ✅ Optimize buffer layouts and shader code

---

## 📚 Code Architecture

### Design Patterns

1. **Separation of Concerns**
   - `viewer.py`: Window management, event handling, render loop
   - `*_geometry.py`: Geometry data and setup
   - Shader files: Rendering logic

2. **Resource Management**
   - RAII pattern: Automatic cleanup in `__del__()` methods
   - OpenGL resources properly released

3. **Uniform Interface**
   - All drawable objects implement: `setup()`, `draw(projection, view, model)`
   - Consistent API across all examples

### Workflow

```
1. Initialize Viewer (window, OpenGL context)
2. Create Geometry Object (Triangle, Cube, etc.)
3. Call object.setup() (VAO/VBO setup, shader compilation, uniform upload)
4. Add object to viewer: viewer.add(object)
5. Run render loop: viewer.run()
```

---

## 🎯 Key Concepts Demonstrated

### Buffer Management
- **Separate Buffers**: Each attribute in its own VBO
- **Interleaved Buffers**: Multiple attributes in one VBO with stride/offset
- **Index Buffers**: EBO for efficient vertex reuse

### Lighting Models
- **Ambient**: Base illumination
- **Diffuse**: Lambertian reflection (N·L)
- **Specular**: Phong reflection (R·V)^n
- **Material Properties**: K_materials (diffuse, specular, ambient coefficients)

### Texture Mapping
- Texture coordinate interpolation
- Multiple textures per object
- Texture + lighting blending

### Camera Control
- Trackball interface for intuitive 3D navigation
- Perspective projection with adaptive clipping planes
- Pan, zoom, rotate operations

---

## 📦 Optional: Run in Google Colab

For headless environments or Google Colab:

```python
!apt install -y xvfb
!pip install PyOpenGL glfw opencv-python pyvirtualdisplay

from pyvirtualdisplay import Display
Display(visible=0, size=(800,600)).start()

# Then run your viewer.py
```

---

## 🚀 Suggested Extensions

### Beginner Projects
- Add keyboard controls to switch between shaders
- Implement simple animations (rotation, translation)
- Create your own geometry (sphere, cylinder, etc.)

### Intermediate Projects
- Load 3D models from `.obj` files
- Implement multiple light sources
- Add shadow mapping
- Create a simple scene with multiple objects

### Advanced Projects
- Implement PBR (Physically Based Rendering)
- Add post-processing effects (bloom, HDR)
- Implement normal mapping
- Create a simple game or interactive scene

---

## 🙋‍♂️ Troubleshooting

### Common Issues

**Black Screen?**
- Check shader compilation errors (printed to console)
- Verify VAO is activated before `glDraw*` calls
- Ensure projection matrix is set correctly
- Check if depth test is enabled for 3D objects

**GLFW Error?**
- Ensure OpenGL 3.3+ is supported
- Update GPU drivers
- Try CPU OpenGL emulation (Mesa on Linux)
- Check if running in headless mode (needs display)

**Import Errors?**
- Make sure you're running from the example directory
- Verify all dependencies are installed
- Check Python version (≥ 3.8)

**Texture Not Loading?**
- Verify texture file path is correct (relative to shader file location)
- Check image format (JPEG/PNG supported via OpenCV)
- Ensure texture unit is properly bound

**Shader Compilation Failed?**
- Check GLSL version (`#version 330 core`)
- Verify attribute locations match between shader and Python code
- Ensure uniform names match exactly

---

## 📖 Additional Resources

- [Learn OpenGL](https://learnopengl.com/) - Comprehensive OpenGL tutorial
- [OpenGL Documentation](https://www.khronos.org/opengl/)
- [GLSL Reference](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)

---

## 👨‍🏫 Instructor Notes

### Course Structure Suggestions

1. **Week 1-2**: Basic setup, triangle with flat/color interpolation shaders
2. **Week 3-4**: 3D transformations, cube example, camera control
3. **Week 5-6**: Lighting models (Phong shading)
4. **Week 7-8**: Texture mapping, advanced techniques

### Assessment Ideas

- **Lab 1**: Modify triangle colors and positions
- **Lab 2**: Create a new geometry (e.g., pyramid, cylinder)
- **Lab 3**: Implement a new shading model
- **Lab 4**: Create a textured scene with multiple objects

### Code Quality

- All code follows Python PEP 8 style guidelines
- Comprehensive comments explaining OpenGL concepts
- Modular design for easy extension
- Error handling for common OpenGL issues

---

## 📝 License

This starter code is provided for educational purposes. Feel free to modify and extend for your courses.

---

## 🤝 Contributing

If you find bugs or have suggestions for improvements, please create an issue or submit a pull request.

---

**Happy Coding! 🎨✨**
