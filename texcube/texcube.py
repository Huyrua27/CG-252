import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import ctypes
import glfw
import numpy as np
import OpenGL.GL as GL


"""
TOP (y =+1): EFGH
                                |
   G (-1, +1, -1)   ......................... H: (+1, +1, -1) 
   color: (0,1,1)   |           |           |    WHITE: (1, 1, 1)
                    |           |           |
                    |           |           |
            --------------------------------------->X
                    |           |           |
                    |           |           |
                    |           |           |
   F: (-1, +1, +1)  ......................... E: (+1, +1, +1)
   BLUE: (0, 0, 1)              |              color: (1,0,1)
                                V 
                                Z

BOTTOM (y=-1): ABCD
                                |
    C: (-1, -1, -1) ......................... D: (+1, -1, -1)
    GREEN: (0,1,0)  |           |           |  color: (1,1,0)
                    |           |           |
                    |           |           |
            --------------------------------------->X
                    |           |           |
                    |           |           |
                    |           |           |
    B: (-1, -1, +1) ......................... A: (+1, -1, +1)
    BLACK: (0,0,0)              |               RED: (1,0,0)
                                V 
                                Z

Texture (2D image: 3x4, see: shape/texcube/image/texture.jpeg
        0             1/4             2/4             3/4             1.0  
   0    ...............................F...............E.......................>X
        |              |               |               |               |
        |              |               |               |               |
        |              |               |               |               |
   1/3  E..............F...............G...............H...............E
        |              |               |               |               |
        |              |               |               |               |
        |              |               |               |               |
   2/3  A..............B...............C...............D...............A
        |              |               |               |               |
        |              |               |               |               |
        |              |               |               |               |
   1.0  ...............................B...............A................
        |
        V 
        Y
"""


class TexCube(object):
    def __init__(self, vert_shader, frag_shader):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.vertices = np.array(
            [
                # YOUR CODE HERE to specify vertices' coordinates
            ],
            dtype=np.float32
        )

        # concatenate three sequences of triangle strip: [0 - 9] [10 - 13] [14-17]
        # => repeat 9, 10, 13, 14
        self.indices = np.array(
            [],  # YOUR CODE HERE to specify indices
            dtype=np.int32
        )

        self.normals = None  # YOUR CODE HERE to compute vertices' normals using the coordinates

        # texture coordinates
        self.texcoords = np.array(
            [
                # YOUR CODE HERE to specify vertices' texture coordinates
            ],
            dtype=np.float32
        )

        self.colors = np.array(
            [
                # YOUR CODE HERE to specify vertices' color
            ],
            dtype=np.float32
        )

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        # setup VAO for drawing cylinder's side
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao.add_vbo(3, self.normals, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing cylinder's side, bottom and top
        self.vao.add_ebo(self.indices)

        # setup textures (only for phong_texture shader)
        if 'phong_texture' in self.frag_shader.lower():
            self.uma.setup_texture("texture", "./image/texture.jpeg")
            phong_factor = 0.0  # blending factor for phong shading and texture
            self.uma.upload_uniform_scalar1f(phong_factor, 'phong_factor')

        # Setup lighting if using Gouraud or Phong shader
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower() and 'texture' not in self.frag_shader.lower():
            self.lighting.setup_phong(mode=1)
        elif 'phong_texture' in self.frag_shader.lower():
            self.lighting.setup_phong(mode=1)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # Setup lighting if using Gouraud or Phong shader (not phong_texture, already setup in setup())
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower() and 'texture' not in self.frag_shader.lower():
            self.lighting.setup_phong(mode=1)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

