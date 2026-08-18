#version 430 core

struct Trig {
    uvec4 points;
    float region;
    uvec4 edges;
};

struct Edge {
    uvec2 Points;
    uint is_boundary;
    uint region;
};


// Speicherpuffer für die globalen FEM-Knotenkoordinaten
layout(std430, binding = 0) readonly buffer NodeBuffer {
    vec4 nodes[];
};

// Triangle Buffer
layout(std430, binding = 1) readonly buffer TriangleBuffer {
    Trig trigs[];
};

// Speicherpuffer für die Element-Konnektivität (Dreiecke)
layout(std430, binding = 2) readonly buffer EdgeBuffer {
    Edge edges[];
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
    // gl_vertexID den lokalen Knoten index, 0, 1, 2
    uint nodeIndex = trigs[gl_InstanceID].points[gl_VertexID];
    // Baryzentrische Koordinate der aktuellen Ecke zuweisen
    vBarycentric = barycentricCoords[gl_VertexID];
        
    vTriangleIndex = uint(gl_InstanceID);
    // Globalen Vertex-Index glatt an den Fragment-Shader durchreichen, wozu?
    vVertexIndex = nodeIndex;
    vec4 position = nodes[nodeIndex];
    gl_Position = u_projection_mat*u_view_mat*u_model_mat*position;
}