#version 420 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

uniform mat4 u_view_mat[6];
uniform mat4 u_projection_mat;
uniform int light_index;

out vec4 FragPos; // FragPos from GS (output per emitvertex)

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        gl_Layer = face+6*light_index; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = u_projection_mat * u_view_mat[face] * FragPos;
            EmitVertex();
        }    
        EndPrimitive();
    }
}



// void emitFace(mat4 m) {
//     for(int i = 0; i < 3; ++i) {
//         FragPos = gl_in[i].gl_Position;
//         gl_Position = u_projection_mat * m * FragPos;
//         EmitVertex();
//     }
//     EndPrimitive();
// }

// void main()
// {
//     gl_Layer = 0;
//     emitFace(u_view_mat[0]);

//     gl_Layer = 1;
//     emitFace(u_view_mat[1]);

//     gl_Layer = 2;
//     emitFace(u_view_mat[2]);

//     gl_Layer = 3;
//     emitFace(u_view_mat[3]);

//     gl_Layer = 4;
//     emitFace(u_view_mat[4]);

//     gl_Layer = 5;
//     emitFace(u_view_mat[5]);
// }