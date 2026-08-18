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

in vec3 vBarycentric;
flat in uint vVertexIndex;
flat in uint vTriangleIndex;

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
    uint edge;
    bool found = false;
    if (vBarycentric.x <= 0.1)
    {
        edge = trigs[vTriangleIndex].edges.y;
        if (edges[edge].is_boundary == 1)
            found = true;
    }
    if (vBarycentric.y <= 0.1)
    {
        edge = trigs[vTriangleIndex].edges.z;
        if (edges[edge].is_boundary == 1)
            found = true;
    }
    if (vBarycentric.z <= 0.1)
    {
        edge = trigs[vTriangleIndex].edges.x;
        if (edges[edge].is_boundary == 1)
            found = true;
    }

    if (found == true)
    {
//        {
            thickness = edgeFactor(5);
            lineColor = vec4(1,0,0,1);
//        }
    }
    // Mischen der Farben: nahe an der Kante dominiert lineColor
    fragColor = mix(lineColor, faceColor, thickness);
}