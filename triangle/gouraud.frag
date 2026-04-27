#version 330 core

precision mediump float;

// Receiving interpolated color from vertex shader
in vec3 colorInterp;

out vec4 fragColor;

void main() {
  // Simply output the interpolated color (lighting was computed per-vertex)
  fragColor = vec4(colorInterp, 1.0);
}
