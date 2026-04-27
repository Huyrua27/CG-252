#version 330 core

uniform vec3 encoded_id_rgb;
uniform int use_texture;
uniform int use_alpha_key;
uniform sampler2D tex0;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    if (use_texture == 1) {
        vec4 texel = texture(tex0, v_uv);
        if (texel.a < 0.2) {
            discard;
        }
        if (use_alpha_key == 1 && texel.a > 0.95 && dot(texel.rgb, vec3(0.3333)) > 0.998) {
            discard;
        }
    }
    frag_color = vec4(encoded_id_rgb, 1.0);
}
