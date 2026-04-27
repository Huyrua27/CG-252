#version 330 core

// input attribute variable, given per vertex
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

uniform mat4 projection, modelview;
uniform mat3 K_materials;
uniform mat3 I_light;
uniform float shininess;
uniform vec3 light_pos;

out vec3 colorInterp;  // Interpolated color (lighting computed per-vertex)

void main(){
  // Transform vertex position to world space
  vec4 vertPos4 = modelview * vec4(position, 1.0);
  vec3 vertPos = vec3(vertPos4) / vertPos4.w;
  
  // Transform normal to world space
  mat4 normal_matrix = transpose(inverse(modelview));
  vec3 N = normalize(vec3(normal_matrix * vec4(normal, 0.0)));
  
  // Compute lighting vectors
  vec3 L = normalize(light_pos - vertPos);
  vec3 R = reflect(-L, N);
  vec3 V = normalize(-vertPos);
  
  // Compute Phong lighting components (per-vertex)
  float NdotL = max(dot(N, L), 0.0);
  float specAngle = max(dot(R, V), 0.0);
  float specular = pow(specAngle, shininess);
  
  // Combine lighting components
  vec3 g = vec3(NdotL, specular, 1.0);  // [diffuse, specular, ambient]
  vec3 lighting = matrixCompMult(K_materials, I_light) * g;
  
  // Combine lighting with vertex color
  colorInterp = 0.5 * lighting + 0.5 * color;
  
  gl_Position = projection * vertPos4;
}
