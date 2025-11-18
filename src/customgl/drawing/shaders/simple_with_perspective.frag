#version 420 core
layout(location = 0) out vec4 FragColor;
in vec2 v_textureuv;
in vec3 v_normal;
uniform sampler2D object_texture;
void main()
{
      FragColor = texture(object_texture, v_textureuv);
}