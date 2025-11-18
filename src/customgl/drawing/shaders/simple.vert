#version 420 core
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_textureuv;
out float y_pos;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(a_position, 1.0);
    y_pos = a_position.y;
    TexCoord = vec2(a_textureuv.x, a_textureuv.y);
}