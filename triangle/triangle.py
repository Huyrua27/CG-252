import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import ctypes

class Triangle:
    def __init__(self, vert_shader, frag_shader):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.vertices = np.array([
            [-1, -1, 0], # A
            [+1, -1, 0], # B
            [ 0, +1, 0]  # C
        ], dtype=np.float32) # numpy: have to specify float32
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.colors = np.array([
            [1.0, 0.0, 0.0], # vertex A: color RED
            [0.0, 1.0, 0.0], # vertex B: GREEN
            [0.0, 0.0, 1.0]  # vertex C: BLUE
        ], dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def setup(self):
        self.vao.add_vbo(0, # index of the attribute in shader
                         self.vertices, # variable in python program
                         ncomponents=3, # x,y,x
                         dtype=GL.GL_FLOAT, # type GL_FLOAT = float32 in numpy (careful)
                         normalized=False,  # normalize vector or not (when normal)
                         stride=0,
                         offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Detect shader type and setup lighting accordingly
        # Gouraud shading computes lighting in vertex shader
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        else:
            # Phong shading computes lighting in fragment shader
            self.lighting.setup_phong(mode=1)
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        self.vao.deactivate()

class TriangleEx:
    def __init__(self, vert_shader, frag_shader):
        """
        self.vertex_attrib:
        each row: v.x, v.y, v.z, c.r, c.g, c.b, n.x, n.y, n.z
                  v.x, v.y, v.z, c.r, c.g, c.b, n.x, n.y, n.z
        =>  (a) stride = nbytes(v0.x -> v1.x) = 9*4 = 36
            (b) offset(vertex) = ctypes.c_void_p(0); can use "None"
                offset(color) = ctypes.c_void_p(3*4)
                offset(normal) = ctypes.c_void_p(6*4)
        """
        vertex_color = np.array([
            [-1, -1, 0, 1.0, 0.0, 0.0],
            [+1, -1, 0, 0.0, 1.0, 0.0],
            [ 0, +1, 0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        self.vertex_attrib = np.concatenate([vertex_color, normals], axis=1)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def setup(self):
        stride = 9*4
        offset_v = ctypes.c_void_p(0)  # None
        offset_c = ctypes.c_void_p(3*4)
        offset_n = ctypes.c_void_p(6*4)
        self.vao.add_vbo(0, self.vertex_attrib, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=offset_v)
        self.vao.add_vbo(1, self.vertex_attrib, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=offset_c)
        self.vao.add_vbo(2, self.vertex_attrib, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=stride, offset=offset_n)

        GL.glUseProgram(self.shader.render_idx)
        normalMat = np.identity(4, 'f')
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Setup Phong lighting using LightingManager
        self.lighting.setup_phong(mode=1)
        return self

    def draw(self, projection, view, model):
        self.vao.activate()
        GL.glUseProgram(self.shader.render_idx)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        self.vao.deactivate()