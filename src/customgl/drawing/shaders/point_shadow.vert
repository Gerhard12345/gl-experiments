#version 420 core
uniform mat4 u_model_mat;
layout(location=0) in vec3 a_position;
void main()
{
    gl_Position = u_model_mat * vec4(a_position, 1.0);
}