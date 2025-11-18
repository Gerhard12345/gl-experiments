#version 420 core
layout(location = 0) out float FragDepth;
void main()
{
   // This is exactly the same, as if using out vec4 myPosition = gl_Position in vertex shader and setting 
   // FragDepth = myPosition*0.5+0.5;
   FragDepth = gl_FragCoord.z;
}
