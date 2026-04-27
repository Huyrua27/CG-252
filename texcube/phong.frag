#version 330 core

precision mediump float;
in vec3 normal_interp;  // Surface normal
in vec3 vert_pos;       // Vertex position
in vec3 color_interp;
in vec2 texcoord_interp;

uniform mat3 K_materials;
uniform mat3 I_light;

uniform int mode;   // Rendering mode
uniform float shininess; // Shininess
uniform vec3 light_pos; // Light position
out vec4 fragColor;

void main() {
  vec3 N = normalize(normal_interp);
  vec3 L = normalize(light_pos - vert_pos);
  vec3 R = reflect(-L, N);      // Reflected light vector
  vec3 V = normalize(-vert_pos); // Vector to viewer

  float specAngle = max(dot(R, V), 0.0);
  float specular = pow(specAngle, shininess);
  vec3 g = vec3(max(dot(L, N), 0.0), specular, 1.0);
  vec3 rgb = 0.5*matrixCompMult(K_materials, I_light) * g + 0.5*color_interp;

  fragColor = vec4(rgb, 1.0);
}
