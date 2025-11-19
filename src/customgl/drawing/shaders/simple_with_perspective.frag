#version 420 core
layout(location = 0) out vec4 FragColor;
in vec2 v_textureuv;
in vec3 v_normal;
uniform sampler2D object_texture;

struct Material {
    sampler2D diffuse;
    sampler2D normal;
    sampler2D ambient_occlusion;
    sampler2D specular;
    float specular_power;
};

uniform Material u_material;

void main()
{
      FragColor = texture(u_material.diffuse, v_textureuv);
}