#version 420 core

in VS_OUT{
    vec2 uv;
    vec3 normal;
    vec3 fragment_position;
    mat3 TBN;
    vec4 fragment_position_in_light_space[N_DIRECTIONAL_LIGHTS];    
    vec3 trig_coords;
} fs_in;


layout(location = 0) out vec4 fragmentcolor;

uniform sampler2DArray directional_shadow_map;
uniform samplerCubeArray depthMap;
uniform vec3 u_viewing_position;
uniform float far_plane;

struct PointLight {
    vec3 position;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};

struct DirectionalLight {
    vec3 direction;
    vec3 diffuse;
    vec3 specular;
};


uniform DirectionalLight u_directional_lights[N_DIRECTIONAL_LIGHTS];
uniform PointLight u_point_lights[N_POINT_LIGHTS];

struct AmbientLight {
    vec3 color;
};
uniform AmbientLight u_ambient_light;

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
    vec4 color = texture(u_material.diffuse, fs_in.uv);
    float meshlines = 0.02;
    if (fs_in.trig_coords.x <= meshlines || fs_in.trig_coords.y <= meshlines || fs_in.trig_coords.z <= meshlines) {
        color = vec4(0,0,0,1);
    }
    else if (fs_in.trig_coords.x <= 1 && fs_in.trig_coords.x <= 1 && fs_in.trig_coords.x <= 1) {
        color = vec4(0,1,0,1);
    }
    float ambient_occlusion = 1.0;//texture(u_material.ambient_occlusion, fs_in.uv).r;
    float specular = texture(u_material.specular, fs_in.uv).r;
    vec3 normal = normalize(fs_in.normal);
    vec3 viewing_direction = normalize(u_viewing_position - fs_in.fragment_position);

    fragmentcolor = vec4(0);
    for (int i=0;i<N_POINT_LIGHTS;i++) {
        vec3 light_to_fragment_vec = u_point_lights[i].position - fs_in.fragment_position;
        vec3 light_to_fragment_direction = normalize(light_to_fragment_vec);
        float light_to_fragment_distance = length(light_to_fragment_vec);
        vec3 halfway_direction = normalize(light_to_fragment_direction + viewing_direction);

        vec3 ambient_intensity = u_ambient_light.color * ambient_occlusion;
        vec3 diffuse_intensity = max(dot(light_to_fragment_direction, normal), 0.0) * u_point_lights[i].diffuse;
        vec3 specular_intensity = specular * pow(max(dot(halfway_direction, normal), 0.0), u_material.specular_power) * u_point_lights[i].specular;

        float attenuation = 1.0 / (u_point_lights[i].constant + u_point_lights[i].linear * light_to_fragment_distance + u_point_lights[i].quadratic * (light_to_fragment_distance * light_to_fragment_distance));
        fragmentcolor += 1.0 / (N_POINT_LIGHTS+N_DIRECTIONAL_LIGHTS) * vec4(attenuation * (ambient_intensity + (diffuse_intensity + specular_intensity)).xyz, 1.0) * color;
    }
    for (int i=0;i<N_DIRECTIONAL_LIGHTS;i++) {
        vec3 light_to_fragment_vec = -u_directional_lights[i].direction;
        vec3 light_to_fragment_direction = normalize(light_to_fragment_vec);
        vec3 halfway_direction = normalize(light_to_fragment_direction + viewing_direction);

        vec3 ambient_intensity = u_ambient_light.color * ambient_occlusion;
        vec3 diffuse_intensity = max(dot(light_to_fragment_direction, normal),0.0) * u_directional_lights[i].diffuse;
        vec3 specular_intensity = specular * pow(max(dot(halfway_direction, normal), 0.0), u_material.specular_power) * u_directional_lights[i].specular;
        
        
        fragmentcolor += 1.0 / (N_POINT_LIGHTS+N_DIRECTIONAL_LIGHTS) * vec4((ambient_intensity + (diffuse_intensity + specular_intensity)).xyz, 1.0) * color;
    }
}