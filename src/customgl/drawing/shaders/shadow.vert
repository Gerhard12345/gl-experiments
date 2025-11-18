#version 420 core
uniform mat4 u_projection_mat;
uniform mat4 u_model_mat;
uniform mat4 u_view_mat;
layout(location=0) in vec3 a_position;
void main()
{
    gl_Position = u_projection_mat*u_view_mat*u_model_mat*vec4(a_position, 1.0);    
}