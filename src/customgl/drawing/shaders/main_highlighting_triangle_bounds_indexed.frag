#version 430 core

struct Trig {
    uvec4 points;
    float region;
    uvec4 edges;
};

struct Edge {
    uvec2 points;
    uint is_boundary;
    uint region;
};

struct Point {
    vec4 position;
    float is_boundary;
};

in vec3 vBarycentric;
in vec2 vBarycentricLine;
flat in uint vEdgeIndex;
flat in uint vTriangleIndex;

uniform int n_vertices;

const int p = 3;
const int COEFFICIENT_COUNT = 3 * p + int((p - 2) * (p - 1) / 2);

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

// Speicherpuffer für die Element-Konnektivität (Dreiecke)
layout(std430, binding = 4) readonly buffer BoundaryEdgesBuffer {
    uint boundary_edges[];
};

layout(std430, binding = 5) readonly buffer CoefficientVectorBuffer {
    float coefficient_vector[][COEFFICIENT_COUNT];
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

vec3 jetColor(float value) {
    float normalizedValue = clamp((value + 300.0) / 600.0, 0.0, 1.0);

    return vec3(
        clamp(1.5 - abs(4.0 * normalizedValue - 3.0), 0.0, 1.0),
        clamp(1.5 - abs(4.0 * normalizedValue - 2.0), 0.0, 1.0),
        clamp(1.5 - abs(4.0 * normalizedValue - 1.0), 0.0, 1.0)
    );
}

void edge_based_polynomials(int edges[2], inout float shape_vector[COEFFICIENT_COUNT])
{
    for(int i=3;i<COEFFICIENT_COUNT;i++)
        shape_vector[i] = 0.0;
}

void compute_shape(out float shape_vector[COEFFICIENT_COUNT])
{
    const int[3][2] edges = {{1, 2}, {0, 2}, {0, 1}};
    shape_vector[0] = vBarycentric.x;
    shape_vector[1] = vBarycentric.y;
    shape_vector[2] = vBarycentric.z;
    if (p == 1)
    {
        return;
    }
    for(int edge = 0;edge<3;edge++)
    {
        edge_based_polynomials(edges[edge], shape_vector);
    }
}


void main() {
    if (n_vertices == 3)
    {
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
        
        float shape_vector[COEFFICIENT_COUNT];
        compute_shape(shape_vector);
        float scalar_product = 0.0;
        for (int i = 0; i < COEFFICIENT_COUNT; ++i) {
           scalar_product += coefficient_vector[vTriangleIndex][i] * shape_vector[i];
        }

        faceColor = vec4(jetColor(scalar_product), 1.0);
        fragColor = mix(lineColor, faceColor, thickness);
    }
    else
    {
        uint is_boundary_edge = edges[vEdgeIndex].is_boundary;
        vec4 lineColor = vec4(0.0, 0.0, 0.0, 1.0);
        if (is_boundary_edge == 1)
        {
            lineColor = vec4(0,0,1,1);
        }
        // Mischen der Farben: nahe an der Kante dominiert lineColor
        fragColor = lineColor;
    }
}