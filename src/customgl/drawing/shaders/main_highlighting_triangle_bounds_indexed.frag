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

const int p = 6;
const int COEFFICIENT_COUNT = 3 * p + int((p - 2) * (p - 1) / 2);
const int COEFFICIENT_COUNT_NODES = 3;
const int COEFFICIENT_COUNT_EDGE = p - 1;

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
    const int n_colors = 920;
    const float maximum_value = 300.0;
    const float minimum_value = -300.0;
    float normalizedValue = clamp((value - minimum_value) / (maximum_value - minimum_value), 0.0, 1.0);
    int colorIndex = min(int(normalizedValue * float(n_colors)), n_colors - 1);
    normalizedValue = float(colorIndex) / float(n_colors - 1);

    return vec3(
        clamp(1.5 - abs(4.0 * normalizedValue - 3.0), 0.0, 1.0),
        clamp(1.5 - abs(4.0 * normalizedValue - 2.0), 0.0, 1.0),
        clamp(1.5 - abs(4.0 * normalizedValue - 1.0), 0.0, 1.0)
    );
}

void jacobi_polynomials(float x, float alpha, inout float edge_shape[p + 1])
{
    for(int i=0;i<p + 1;i++)
        edge_shape[i] = 0.0;

    edge_shape[0] = 1;
    if(p == 0)
        return;
    edge_shape[1] = 0.5 * (alpha + (alpha + 2) * x);
    float alpha2 = alpha*alpha;
    for(int j=1;j<p;j++)
    {
        float a_1 = (2 * j + alpha + 1) / ((2 * j + 2) * (j + alpha + 1) * (2 * j + alpha));
        float a_2 = (2 * j + alpha + 2) * (2 * j + alpha);
        float a_3 = j * (j + alpha) * (2 * j + alpha + 2) / ((j + 1) * (j + alpha + 1) * (2 * j + alpha));
        edge_shape[j + 1] = a_1 * (a_2 * x + alpha2) * edge_shape[j] - a_3 * edge_shape[j - 1];
    }
}

void integrated_jacobi_polynomials(float x, float alpha, out float integrated_jacobi_polynomial[p + 1])
{
    for(int i=0;i<p + 1;i++)
        integrated_jacobi_polynomial[i] = 0.0;
    
    integrated_jacobi_polynomial[0] = 1;
    if(p == 0)
        return;
    integrated_jacobi_polynomial[1] = x + 1;
    if(p == 1)
        return;
    float jacobi_poly_vals[p+1];
    jacobi_polynomials(x, alpha, jacobi_poly_vals);
    for(int  j=2;j<p+1;j++)
    {
        float a_1 = (2 * j + 2 * alpha) / ((2 * j + alpha - 1) * (2 * j + alpha));
        float a_2 = 2 * alpha / ((2 * j + alpha - 2) * (2 * j + alpha));
        float a_3 = (2 * j - 2) / ((2 * j + alpha - 1) * (2 * j + alpha - 2));
        integrated_jacobi_polynomial[j] = a_1 * jacobi_poly_vals[j] + a_2 * jacobi_poly_vals[j - 1] - a_3 * jacobi_poly_vals[j - 2];
    }
}

void edge_based_polynomials(int edges[2], int first_edge_coefficient_for_current_edge, inout float shape_vector[COEFFICIENT_COUNT])
{
    float l1 = vBarycentric[edges[1]] + vBarycentric[edges[0]];
    float l2 = vBarycentric[edges[1]] - vBarycentric[edges[0]];
    if (trigs[vTriangleIndex].points[edges[0]] > trigs[vTriangleIndex].points[edges[1]])
    {
        l2 = -l2;
    }
    float x_eval = l2 / l1;
    float integrated_jacobi_polynomial[p + 1];
    integrated_jacobi_polynomials(x_eval, 0.0, integrated_jacobi_polynomial);
    for(int i=0; i < p - 1; i++)
    {
        shape_vector[first_edge_coefficient_for_current_edge + i] = integrated_jacobi_polynomial[i + 2] * pow(l1, i+2);
    }
}

void bubble_functions(int first_bubble_coefficient, inout float shape_vector[COEFFICIENT_COUNT])
{
    if (p < 3)
    {
        return;
    }

    float l1 = 2.0 * vBarycentric[0] - 1.0;
    int bubble_coefficient = first_bubble_coefficient;
    for (int i = 2; i < p; i++)
    {
        int bubble_order = p - i;
        float h_values[p + 1];
        integrated_jacobi_polynomials(l1, float(2 * i - 1), h_values);
        float edge_factor = shape_vector[3 + i - 2];
        for (int j = 0; j < bubble_order; j++)
        {
            shape_vector[bubble_coefficient + j] = edge_factor * h_values[j + 1];
        }
        bubble_coefficient += bubble_order;
    }
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
    int first_edge_dof = COEFFICIENT_COUNT_NODES;
    for(int edge = 0;edge<3;edge++)
    {
        edge_based_polynomials(edges[edge], first_edge_dof, shape_vector);
        first_edge_dof += COEFFICIENT_COUNT_EDGE;
    }
    bubble_functions(3 * p, shape_vector);
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