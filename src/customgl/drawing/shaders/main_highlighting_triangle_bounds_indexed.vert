#version 430 core

// Speicherpuffer für die globalen FEM-Knotenkoordinaten
layout(std430, binding = 0) readonly buffer NodeBuffer {
    vec2 nodes[];
};

// Speicherpuffer für die Element-Konnektivität (Dreiecke)
layout(std430, binding = 1) readonly buffer ElementBuffer {
    uvec4 elements[];
};

// Speicherpuffer für die Element-Konnektivität (Dreiecke)
layout(std430, binding = 2) readonly buffer EdgeBuffer {
    uvec4 edges[];
};


// Ausgabevariablen für den Fragment-Shader
out vec3 vBarycentric;
flat out uint vVertexIndex;
flat out uint vTriangleIndex;

uniform mat4 u_projection_mat;
uniform mat4 u_model_mat;
uniform mat4 u_view_mat;

// Implizites Referenzdreieck für die baryzentrischen Koordinaten
const vec3 barycentricCoords[3] = vec3[3](
    vec3(1.0, 0.0, 0.0), // Ecke 0
    vec3(0.0, 1.0, 0.0), // Ecke 1
    vec3(0.0, 0.0, 1.0)  // Ecke 2
);

void main() {
    // gl_InstanceID bestimmt, welches FEM-Element gezeichnet wird
    uvec3 currentElement = elements[gl_InstanceID].xyz;
    
    // gl_VertexID läuft pro Instanz von 0 bis 2
    uint nodeIndex;
    if (gl_VertexID == 0) nodeIndex = currentElement.x;
    else if (gl_VertexID == 1) nodeIndex = currentElement.y;
    else nodeIndex = currentElement.z;
    
    // Baryzentrische Koordinate der aktuellen Ecke zuweisen
    vBarycentric = barycentricCoords[gl_VertexID];
    
    // Globalen Vertex-Index glatt an den Fragment-Shader durchreichen
    vVertexIndex = nodeIndex;
    vTriangleIndex = uint(gl_InstanceID);
    
    // Position auslesen und transformieren
    vec2 position = nodes[nodeIndex];
    vec4 position4 = vec4(position, 0.0, 1.0);
    gl_Position = u_projection_mat*u_view_mat*u_model_mat*position4;
}