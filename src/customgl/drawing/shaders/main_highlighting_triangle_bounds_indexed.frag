#version 430 core

in vec3 vBarycentric;
flat in uint vVertexIndex;
flat in uint vTriangleIndex;

layout(std430, binding = 3) readonly buffer HighlightBuffer {
    vec4 highlights[];
};

out vec4 fragColor;

// Berechnet die adaptive Kanten-Glättung basierend auf der Bildschirm-Änderungsrate
float edgeFactor() {
    // fwidth berechnet die Änderung pro Pixel (Größe des Dreiecks ist implizit drin)
    vec3 d = fwidth(vBarycentric);
    
    // smoothstep erzeugt einen weichen Übergang (Antialiasing), w kontrolliert die Linienbreite
    float w = 0.85; // Breite der Kante in Bildschirmkoordinaten
    vec3 a3 = smoothstep(vec3(0.0), d * w, vBarycentric);
    
    return min(min(a3.x, a3.y), a3.z);
}

void main() {
    float thickness = edgeFactor();
    
    // Grundfarbe des FEM-Elements (Dunkelgrau)
    vec4 faceColor = vec4(0.18, 1.0, 0.18, 1.0);
    
    // Linienfarbe des Gitters (Cyan)
    vec4 lineColor = vec4(0.0, 0.0, 0.0, 1.0);
    
    // Sonderlogik basierend auf dem Highlight-Buffer:
    if (highlights[vTriangleIndex].x > 0.5) {
        faceColor = vec4(0.8, 0.15, 0.15, 1.0);
    }
    
    // Mischen der Farben: nahe an der Kante dominiert lineColor
    fragColor = mix(lineColor, faceColor, thickness);
}