#version 420 core

in VS_OUT{
    vec2 uv;
    vec3 normal;
    vec3 fragment_position;
    mat3 TBN;
    vec4 fragment_position_in_light_space[N_DIRECTIONAL_LIGHTS];    
} fs_in;


layout(location = 0) out vec4 fragmentcolor;

uniform sampler2DArray directional_shadow_map;
uniform samplerCubeArray depthMap;
uniform vec3 u_viewing_position;
uniform float far_plane;

struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

struct DirectionalLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};


uniform DirectionalLight u_directional_lights[N_DIRECTIONAL_LIGHTS];
uniform PointLight u_point_lights[N_POINT_LIGHTS];

struct Material {
    sampler2D diffuse;
    sampler2D normal;
    sampler2D ambient_occlusion;
    sampler2D specular;
    float specular_power;
};

uniform Material u_material;


float is_in_shadow(vec4 fragment_position_in_light_space[N_DIRECTIONAL_LIGHTS], vec3 normal, int component)
{
    float bias = max(0.05 * (1.0 - dot(normal, u_directional_lights[component].direction)), 0.005);
    vec4 light_space_pos = fragment_position_in_light_space[component];
    vec2 texelSize = 1.0 / textureSize(directional_shadow_map, 0).xy;
    light_space_pos /= light_space_pos.w;
    light_space_pos = light_space_pos*0.5+0.5;
    float current_depth = light_space_pos.z;
    float shadow = 0.0;
    for(int x=-1;x<=1;x++) {
        for(int y=-1;y<=1;y++) {
            float mindepth = texture(directional_shadow_map, vec3(light_space_pos.xy+vec2(x,y)*texelSize,component)).r;
            shadow += mindepth < current_depth-bias ? 1.0 : 0.0;
        }
    }
    return shadow/9.0;
}


float is_in_point_shadow(vec3 fragPos, vec3 normal, int component)
{
    float bias = max(0.5 * (1.0 - dot(normal, u_point_lights[component].position)), 0.015);
    vec3 fragToLight = fragPos - u_point_lights[component].position;
    float currentDepth = length(fragToLight);
    int samples  = 20;
    float diskRadius = 0.05;
    vec3 sampleOffsetDirections[20] = vec3[]
    (
        vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1), 
        vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
        vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
        vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
        vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
    );
    float shadow = 0.0;
    for(int i = 0; i < samples; ++i)
    {
        float closestDepth = texture(depthMap, vec4(fragToLight + sampleOffsetDirections[i] * diskRadius, component)).r;
        closestDepth *= far_plane;
        if(currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);
    // float closestDepth = 0.0;
    // for(int i=0;i<4;i++)
    //     closestDepth += texture(depthMap, vec4(fragToLight, 2)).r;
    // fragmentcolor = vec4(vec3(closestDepth / far_plane), 1.0);
    return shadow;
}



void main()
{
    vec4 color = texture(u_material.diffuse, fs_in.uv);
    float ambient_occlusion = texture(u_material.ambient_occlusion, fs_in.uv).r;
    float specular = texture(u_material.specular, fs_in.uv).r;
    vec3 normal = texture(u_material.normal, fs_in.uv).rgb;
    normal = normal * 2.0 - 1.0;   
    normal = normalize(fs_in.TBN * normal);
    vec3 viewing_direction = normalize(u_viewing_position - fs_in.fragment_position);

    fragmentcolor = vec4(0);
    for (int i=0;i<N_POINT_LIGHTS;i++) {
        vec3 light_to_fragment_vec = u_point_lights[i].position - fs_in.fragment_position;
        vec3 light_to_fragment_direction = normalize(light_to_fragment_vec);
        float light_to_fragment_distance = length(light_to_fragment_vec);
        vec3 halfway_direction = normalize(light_to_fragment_direction + viewing_direction);

        vec3 ambient_intensity = u_point_lights[i].ambient * ambient_occlusion;
        vec3 diffuse_intensity = max(dot(light_to_fragment_direction, normal), 0.0) * u_point_lights[i].diffuse;
        vec3 specular_intensity = specular * pow(max(dot(halfway_direction, normal), 0.0), u_material.specular_power) * u_point_lights[i].specular;
        
        float shadow = is_in_point_shadow(fs_in.fragment_position, normal, i);
        float attenuation = 1.0 / (u_point_lights[i].constant + u_point_lights[i].linear * light_to_fragment_distance + u_point_lights[i].quadratic * (light_to_fragment_distance * light_to_fragment_distance));
        fragmentcolor += 1.0 / (N_POINT_LIGHTS) * vec4(attenuation * (ambient_intensity + (1.0-shadow) * (diffuse_intensity + specular_intensity)).xyz, 1.0) * color;
    }
    for (int i=0;i<N_DIRECTIONAL_LIGHTS;i++) {
        vec3 light_to_fragment_vec = -u_directional_lights[i].direction;
        vec3 light_to_fragment_direction = normalize(light_to_fragment_vec);
        vec3 halfway_direction = normalize(light_to_fragment_direction + viewing_direction);

        vec3 ambient_intensity = u_directional_lights[i].ambient * ambient_occlusion;
        vec3 diffuse_intensity = max(dot(light_to_fragment_direction, normal),0.0) * u_directional_lights[i].diffuse;
        vec3 specular_intensity = specular * pow(max(dot(halfway_direction, normal), 0.0), u_material.specular_power) * u_directional_lights[i].specular;
        
        float shadow = is_in_shadow(fs_in.fragment_position_in_light_space, -normal, i);
        fragmentcolor += 1.0 / (N_POINT_LIGHTS+N_DIRECTIONAL_LIGHTS) * vec4((ambient_intensity + (1-shadow) * (diffuse_intensity + specular_intensity)).xyz, 1.0) * color;
    }
}