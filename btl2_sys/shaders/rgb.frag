#version 330 core

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in vec4 v_light_space_pos;

uniform vec3 camera_pos;
uniform vec3 sun_direction;
uniform vec3 sun_color;
uniform float sun_intensity;
uniform vec3 fill_position;
uniform vec3 fill_color;
uniform float fill_intensity;
uniform vec3 sky_color;
uniform float ambient_strength;
uniform int street_light_count;
uniform vec3 street_light_positions[16];
uniform vec3 street_light_color;
uniform float street_light_intensity;
uniform int shadow_enabled;
uniform float shadow_strength;
uniform float shadow_bias;

uniform vec3 material_kd;
uniform vec3 material_ks;
uniform vec3 material_ka;
uniform float material_shininess;
uniform int use_texture;
uniform int use_alpha_key;
uniform sampler2D tex0;
uniform sampler2D shadow_map;
uniform vec2 uv_offset;

out vec4 frag_color;

float compute_shadow_factor(vec3 normal_vec, vec3 sun_L) {
    if (shadow_enabled == 0) {
        return 1.0;
    }

    vec3 proj = v_light_space_pos.xyz / max(v_light_space_pos.w, 1e-6);
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
        return 1.0;
    }

    float current_depth = proj.z;
    float bias = max(shadow_bias * (1.0 - max(dot(normal_vec, sun_L), 0.0)), shadow_bias * 0.35);

    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float closest_depth = texture(shadow_map, proj.xy + vec2(x, y) * texel_size).r;
            lit += (current_depth - bias <= closest_depth) ? 1.0 : 0.0;
        }
    }
    float visibility = lit / 9.0;
    return mix(1.0, visibility, clamp(shadow_strength, 0.0, 1.0));
}

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(camera_pos - v_world_pos);
    if (dot(N, V) < 0.0) {
        // Thin meshes (leaves, cloth) are often single-sided in OBJ exports.
        // Flip normals toward camera to avoid unrealistically black backfaces.
        N = -N;
    }

    vec3 albedo = material_kd;
    float alpha = 1.0;
    if (use_texture == 1) {
        vec4 texel = texture(tex0, v_uv + uv_offset);
        if (texel.a < 0.2) {
            discard;
        }
        albedo *= texel.rgb;
        alpha = texel.a;
        if (use_alpha_key == 1 && texel.a > 0.95 && dot(texel.rgb, vec3(0.3333)) > 0.998) {
            discard;
        }
    }

    vec3 sun_L = normalize(-sun_direction);
    float sun_diff = max(dot(N, sun_L), 0.0);
    vec3 sun_H = normalize(sun_L + V);
    float sun_spec = pow(max(dot(N, sun_H), 0.0), material_shininess);

    vec3 fill_L_vec = fill_position - v_world_pos;
    float fill_dist = max(length(fill_L_vec), 1e-4);
    vec3 fill_L = fill_L_vec / fill_dist;
    float fill_att = 1.0 / (1.0 + 0.07 * fill_dist + 0.012 * fill_dist * fill_dist);
    float fill_diff = max(dot(N, fill_L), 0.0) * fill_att;
    vec3 fill_H = normalize(fill_L + V);
    float fill_spec = pow(max(dot(N, fill_H), 0.0), material_shininess) * fill_att;

    float sun_shadow = compute_shadow_factor(N, sun_L);
    vec3 sun_term = sun_color * max(sun_intensity, 0.0);
    vec3 fill_term = fill_color * max(fill_intensity, 0.0);

    vec3 street_diffuse = vec3(0.0);
    vec3 street_specular = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        if (i >= street_light_count) {
            break;
        }
        vec3 L_vec = street_light_positions[i] - v_world_pos;
        float dist = max(length(L_vec), 1e-4);
        vec3 L = L_vec / dist;

        // Point-light falloff tuned for road-scale lamps.
        float att = 1.0 / (1.0 + 0.055 * dist + 0.018 * dist * dist);

        // Strengthen illumination on road/object surfaces below the lamp.
        float vertical = clamp((street_light_positions[i].y - v_world_pos.y) / 8.0, 0.0, 1.0);
        float diff = max(dot(N, L), 0.0) * att * vertical;

        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), material_shininess) * att * vertical;

        vec3 street_term = street_light_color * max(street_light_intensity, 0.0);
        street_diffuse += albedo * diff * street_term;
        street_specular += material_ks * spec * street_term * 0.45;
    }

    vec3 ambient_base = mix(material_ka, albedo, 0.55);
    vec3 ambient = ambient_strength * ambient_base * (0.65 + 0.35 * sun_term);
    vec3 diffuse = albedo * (sun_diff * sun_term * sun_shadow + fill_diff * fill_term) + street_diffuse;
    vec3 specular = material_ks * (sun_spec * sun_term * sun_shadow + fill_spec * fill_term) * 0.65 + street_specular;
    vec3 color = ambient + diffuse + specular;

    float view_dist = length(camera_pos - v_world_pos);
    float haze = smoothstep(45.0, 140.0, view_dist) * 0.35;
    color = mix(color, sky_color, haze);

    color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));

    frag_color = vec4(color, alpha);
}
