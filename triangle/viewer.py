import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean windows system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import sys
import os

# Add parent directory to path to import libs
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import triangle
from triangle import Triangle, TriangleEx


# ------------  Viewer class & windows management ------------------------------
class Viewer:
    """ GLFW viewer windows, with classic initialization & graphics loop """
    def __init__(self, width=640, height=480):

        # version hints: create GL windows with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.5, 0.5, 0.5, 0.1)

        # initially empty list of object to draw
        self.drawables = []

    def run(self):
        """ Main render loop for this OpenGL windows """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(None, None, None)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this windows """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            for drawable in self.drawables:
                if hasattr(drawable, 'key_handler'):
                    drawable.key_handler(key)


# -------------- main program and scene setup --------------------------------
def main():
    """ create windows, add shaders & scene objects, then run rendering loop """
    viewer = Viewer()


    # patch = cylinder.TexturedCylinder("./patch/textured/phong_texture.vert", "./patch/textured/phong_texture.frag").setup()

    # Triangles
    #model = Triangle("./color_interp.vert", "./color_interp.frag").setup()
    #model = Triangle("./phong.vert", "./phong.frag").setup()
    #model = Triangle("./gouraud.vert", "./gouraud.frag").setup()

    model = TriangleEx("./phong.vert", "./phong.frag").setup()

    viewer.add(model)

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize windows system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
