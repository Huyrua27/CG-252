#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec2 in_uv;

uniform mat4 light_space_matrix;
uniform mat4 model;

out vec2 v_uv;

void main() {
    vec4 world = model * vec4(in_position, 1.0);
    v_uv = in_uv;
    gl_Position = light_space_matrix * world;
}
