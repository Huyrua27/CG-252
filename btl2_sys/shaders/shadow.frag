#version 330 core

uniform int use_texture;
uniform int use_alpha_key;
uniform sampler2D tex0;
uniform vec2 uv_offset;

in vec2 v_uv;

void main() {
    if (use_texture == 1) {
        vec4 texel = texture(tex0, v_uv + uv_offset);
        if (texel.a < 0.2) {
            discard;
        }
        if (use_alpha_key == 1 && texel.a > 0.95 && dot(texel.rgb, vec3(0.3333)) > 0.998) {
            discard;
        }
    }
}
