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

struct Point {
    vec4 position;
    float is_boundary;
};

in vec3 vBarycentric;
flat in uint vVertexIndex;
flat in uint vTriangleIndex;


// Speicherpuffer für die globalen FEM-Knotenkoordinaten
layout(std430, binding = 0) readonly buffer NodeBuffer {
    Point nodes[];
};

// Triangle Buffer
layout(std430, binding = 1) readonly buffer TriangleBuffer {
    Trig trigs[];
};

// Speicherpuffer für die Element-Konnektivität (Dreiecke)
layout(std430, binding = 2) readonly buffer EdgeBuffer {
    Edge edges[];
};

layout(std430, binding = 3) readonly buffer HighlightBuffer {
    vec4 highlights[];
};

out vec4 fragColor;

// Berechnet die adaptive Kanten-Glättung basierend auf der Bildschirm-Änderungsrate
float edgeFactor(float w) {
    // fwidth berechnet die Änderung pro Pixel (Größe des Dreiecks ist implizit drin)
    vec3 d = fwidth(vBarycentric);
    
    // smoothstep erzeugt einen weichen Übergang (Antialiasing), w kontrolliert die Linienbreite
    vec3 a3 = smoothstep(vec3(0.0), d * w, vBarycentric);
    
    return min(min(a3.x, a3.y), a3.z);
}

int argmin(vec3 values) {
    if (values.x <= values.y && values.x <= values.z) return 0;
    if (values.y <= values.z) return 1;
    return 2;
}

int argmax(vec3 values) {
    if (values.x >= values.y && values.x >= values.z) return 0;
    if (values.y >= values.z) return 1;
    return 2;
}

void main() {
    float w = 0.85; // Breite der Kante in Bildschirmkoordinaten
    float thickness = edgeFactor(w);
    
    // Grundfarbe des FEM-Elements (Dunkelgrau)
    vec4 faceColor = vec4(0.18, 1.0, 0.18, 1.0);
    float factor = 1;

    // Linienfarbe des Gitters (Cyan)
    vec4 lineColor = vec4(0.0, 0.0, 0.0, 1.0);
    
    // Sonderlogik basierend auf dem Highlight-Buffer:
    if (highlights[vTriangleIndex].x > 0.5) {
        faceColor = vec4(0.8, 0.15, 0.15, 1.0);
    }    
    bool found = false;
    int min_barycentric = argmin(vBarycentric);
    int max_barycentric = argmax(vBarycentric);
    uint is_boundary_edge = edges[trigs[vTriangleIndex].edges[min_barycentric]].is_boundary;
    float is_boundary_point = nodes[trigs[vTriangleIndex].points[max_barycentric]].is_boundary;
    if ((vBarycentric[min_barycentric] <= 0.15 &&  (is_boundary_edge == 1)) ||
        (vBarycentric[max_barycentric] >= 0.99 &&  (is_boundary_point == 1)))
    {
        thickness = edgeFactor(7);
        lineColor = vec4(0,0,1,1);

    }
    // Mischen der Farben: nahe an der Kante dominiert lineColor
    fragColor = mix(lineColor, faceColor, thickness);
}