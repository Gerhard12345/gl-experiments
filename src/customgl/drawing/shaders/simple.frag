#version 420 core
layout(location = 0) out vec4 FragColor;
in float y_pos;
in vec2 TexCoord;

uniform sampler2D scene_texture;
uniform sampler2DArray shadow_texture;
uniform int shadow_component;
void main()
{
      if(shadow_component>=0)
      {
            float depth = texture(shadow_texture, vec3(TexCoord, shadow_component)).r;
            FragColor = vec4(depth, depth, depth,1);
      }
      else
            FragColor = texture(scene_texture, TexCoord);
}