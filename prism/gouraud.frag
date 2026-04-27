#version 330 core

precision mediump float;

// Receiving interpolated color from vertex shader
in vec3 colorInterp;

out vec4 out_color;

void main() {
  // Simply output the interpolated color (lighting was computed per-vertex)
  out_color = vec4(colorInterp, 1.0);
}
