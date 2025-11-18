#version 420 core
uniform mat4 u_projection_mat = mat4(1.0);
uniform mat4 u_model_mat = mat4(1.0);
uniform mat4 u_view_mat = mat4(1.0);

layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_textureuv;

out vec2 v_textureuv;
out vec3 v_normal;

void main()
{
    gl_Position = u_projection_mat*u_view_mat*u_model_mat*vec4(a_position, 1.0);
    v_textureuv = a_textureuv;
    v_normal = mat3(transpose(inverse(u_model_mat))) * a_normal;
}