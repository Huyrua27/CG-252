#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 light_space_matrix;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;
out vec4 v_light_space_pos;

void main() {
    vec4 world = model * vec4(in_position, 1.0);
    mat3 normal_mat = transpose(inverse(mat3(model)));

    v_world_pos = world.xyz;
    v_world_normal = normalize(normal_mat * in_normal);
    v_uv = in_uv;
    v_light_space_pos = light_space_matrix * world;

    gl_Position = projection * view * world;
}

