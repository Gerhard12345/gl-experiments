#version 420 core
uniform mat4 u_projection_mat;
uniform mat4 u_model_mat;
uniform mat4 u_view_mat;
uniform mat4 u_view_mat_lightspace[N_DIRECTIONAL_LIGHTS];
uniform mat4 u_projection_mat_lightspace;
uniform vec2 u_texturescale;


layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_textureuv;
layout(location=3) in vec3 a_trig_coords;
layout(location=4) in vec3 a_unused;

out VS_OUT{
    vec2 uv;
    vec3 normal;
    vec3 fragment_position;
    mat3 TBN;
    vec4 fragment_position_in_light_space[N_DIRECTIONAL_LIGHTS];
    vec3 trig_coords;
} vs_out;

void main()
{
    vec3 T = normalize(vec3(u_model_mat * vec4(a_trig_coords,   0.0)));
    vec3 B = normalize(vec3(u_model_mat * vec4(a_unused, 0.0)));
    vec3 N = normalize(vec3(u_model_mat * vec4(a_normal,    0.0)));
    vs_out.TBN = mat3(T, B, N);
    vs_out.normal = mat3(transpose(inverse(u_model_mat))) * a_normal;
    vs_out.uv = a_textureuv * u_texturescale;
    vs_out.trig_coords = a_trig_coords;
    for(int i=0;i<N_DIRECTIONAL_LIGHTS;i++)
        vs_out.fragment_position_in_light_space[i] = u_projection_mat_lightspace*u_view_mat_lightspace[i]*u_model_mat*vec4(a_position, 1.0);

    gl_Position = u_model_mat*vec4(a_position, 1.0);    
    vs_out.fragment_position = gl_Position.xyz;
    gl_Position = u_projection_mat*u_view_mat*gl_Position;
}