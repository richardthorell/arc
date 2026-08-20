import type { ShaderSymbol } from '../ShaderLanguage';

const type = (name: string, description: string, signatures?: ShaderSymbol['signatures']): ShaderSymbol => ({
  name,
  kind: 'type',
  description,
  signatures,
});

const fn = (name: string, description: string, signatures: readonly string[]): ShaderSymbol => ({
  name,
  kind: 'function',
  description,
  signatures: signatures.map((label) => ({ label })),
});

const variable = (name: string, description: string): ShaderSymbol => ({
  name,
  kind: 'variable',
  description,
});

export const glslSymbols: readonly ShaderSymbol[] = [
  type('void', 'Represents the absence of a value.'),
  type('bool', 'Boolean scalar type.'),
  type('int', 'Signed integer scalar type.'),
  type('uint', 'Unsigned integer scalar type.'),
  type('float', 'Single-precision floating-point scalar type.'),
  type('double', 'Double-precision floating-point scalar type.'),
  type('vec2', 'Two-component floating-point vector.', [
    { label: 'vec2(float scalar)' },
    { label: 'vec2(float x, float y)' },
  ]),
  type('vec3', 'Three-component floating-point vector.', [
    { label: 'vec3(float scalar)' },
    { label: 'vec3(float x, float y, float z)' },
    { label: 'vec3(vec2 xy, float z)' },
  ]),
  type('vec4', 'Four-component floating-point vector.', [
    { label: 'vec4(float scalar)' },
    { label: 'vec4(float x, float y, float z, float w)' },
    { label: 'vec4(vec3 xyz, float w)' },
  ]),
  type('ivec2', 'Two-component signed integer vector.'),
  type('ivec3', 'Three-component signed integer vector.'),
  type('ivec4', 'Four-component signed integer vector.'),
  type('uvec2', 'Two-component unsigned integer vector.'),
  type('uvec3', 'Three-component unsigned integer vector.'),
  type('uvec4', 'Four-component unsigned integer vector.'),
  type('bvec2', 'Two-component Boolean vector.'),
  type('bvec3', 'Three-component Boolean vector.'),
  type('bvec4', 'Four-component Boolean vector.'),
  type('mat2', '2 x 2 floating-point matrix.'),
  type('mat3', '3 x 3 floating-point matrix.'),
  type('mat4', '4 x 4 floating-point matrix.', [
    { label: 'mat4(float diagonal)' },
    { label: 'mat4(vec4 c0, vec4 c1, vec4 c2, vec4 c3)' },
  ]),
  type('sampler2D', 'Handle used to sample a two-dimensional texture.'),
  type('sampler2DArray', 'Handle used to sample a two-dimensional texture array.'),
  type('samplerCube', 'Handle used to sample a cube texture.'),
  type('sampler2DShadow', 'Two-dimensional depth-comparison texture sampler.'),
  type('image2D', 'Read/write handle for a two-dimensional image.'),
  type('uimage2D', 'Read/write handle for a two-dimensional unsigned integer image.'),

  fn('radians', 'Converts an angle from degrees to radians.', ['genFType radians(genFType degrees)']),
  fn('degrees', 'Converts an angle from radians to degrees.', ['genFType degrees(genFType radians)']),
  fn('sin', 'Returns the sine of an angle in radians.', ['genFType sin(genFType angle)']),
  fn('cos', 'Returns the cosine of an angle in radians.', ['genFType cos(genFType angle)']),
  fn('tan', 'Returns the tangent of an angle in radians.', ['genFType tan(genFType angle)']),
  fn('pow', 'Raises each component of x to the corresponding component of y.', [
    'genFType pow(genFType x, genFType y)',
  ]),
  fn('exp', 'Returns e raised to each component of x.', ['genFType exp(genFType x)']),
  fn('log', 'Returns the natural logarithm of each component of x.', ['genFType log(genFType x)']),
  fn('sqrt', 'Returns the square root of each component of x.', ['genFType sqrt(genFType x)']),
  fn('inversesqrt', 'Returns the reciprocal square root of each component of x.', ['genFType inversesqrt(genFType x)']),
  fn('abs', 'Returns the absolute value of each component.', ['genType abs(genType x)']),
  fn('floor', 'Rounds each component down to the nearest integer value.', ['genFType floor(genFType x)']),
  fn('ceil', 'Rounds each component up to the nearest integer value.', ['genFType ceil(genFType x)']),
  fn('fract', 'Returns the fractional part of each component.', ['genFType fract(genFType x)']),
  fn('mod', 'Returns x modulo y component-wise.', [
    'genFType mod(genFType x, genFType y)',
    'genFType mod(genFType x, float y)',
  ]),
  fn('min', 'Returns the smaller value component-wise.', [
    'genType min(genType x, genType y)',
    'genType min(genType x, scalar y)',
  ]),
  fn('max', 'Returns the larger value component-wise.', [
    'genType max(genType x, genType y)',
    'genType max(genType x, scalar y)',
  ]),
  fn('clamp', 'Constrains x to the inclusive range between minVal and maxVal.', [
    'genType clamp(genType x, genType minVal, genType maxVal)',
    'genType clamp(genType x, scalar minVal, scalar maxVal)',
  ]),
  fn('mix', 'Linearly interpolates between x and y.', [
    'genType mix(genType x, genType y, genType a)',
    'genType mix(genType x, genType y, float a)',
  ]),
  fn('step', 'Returns 0 when x is below edge and 1 otherwise, component-wise.', [
    'genType step(genType edge, genType x)',
    'genType step(float edge, genType x)',
  ]),
  fn('smoothstep', 'Performs smooth Hermite interpolation between two edges.', [
    'genType smoothstep(genType edge0, genType edge1, genType x)',
  ]),
  fn('length', 'Returns the Euclidean length of a vector.', ['float length(genFType x)']),
  fn('distance', 'Returns the distance between two points.', ['float distance(genFType p0, genFType p1)']),
  fn('dot', 'Returns the dot product of two vectors.', ['float dot(genFType x, genFType y)']),
  fn('cross', 'Returns the cross product of two three-component vectors.', ['vec3 cross(vec3 x, vec3 y)']),
  fn('normalize', 'Returns a vector with the same direction as x and a length of 1.', [
    'genFType normalize(genFType x)',
  ]),
  fn('faceforward', 'Orients a normal to face away from the reference direction.', [
    'genFType faceforward(genFType N, genFType I, genFType Nref)',
  ]),
  fn('reflect', 'Returns the reflection direction for an incident vector and normal.', [
    'genFType reflect(genFType I, genFType N)',
  ]),
  fn('refract', 'Returns the refraction direction for an incident vector, normal, and index ratio.', [
    'genFType refract(genFType I, genFType N, float eta)',
  ]),
  fn('transpose', 'Returns the transpose of a matrix.', ['mat transpose(mat m)']),
  fn('determinant', 'Returns the determinant of a square matrix.', ['float determinant(mat m)']),
  fn('inverse', 'Returns the inverse of a square matrix.', ['mat inverse(mat m)']),
  fn('dFdx', 'Returns the partial derivative of an expression with respect to window x.', [
    'genFType dFdx(genFType p)',
  ]),
  fn('dFdy', 'Returns the partial derivative of an expression with respect to window y.', [
    'genFType dFdy(genFType p)',
  ]),
  fn('fwidth', 'Returns abs(dFdx(p)) + abs(dFdy(p)).', ['genFType fwidth(genFType p)']),
  fn('textureSize', 'Returns the dimensions of a texture at the requested mip level.', [
    'ivec textureSize(gsampler sampler, int lod)',
  ]),
  fn('texture', 'Samples a texture using normalized texture coordinates.', [
    'gvec4 texture(gsampler2D sampler, vec2 P)',
    'gvec4 texture(gsamplerCube sampler, vec3 P)',
  ]),
  fn('textureLod', 'Samples a texture using an explicit level of detail.', [
    'gvec4 textureLod(gsampler2D sampler, vec2 P, float lod)',
  ]),
  fn('texelFetch', 'Fetches a single texel using integer coordinates.', [
    'gvec4 texelFetch(gsampler2D sampler, ivec2 P, int lod)',
  ]),

  variable('gl_Position', 'Vertex-stage output containing the homogeneous clip-space position.'),
  variable('gl_PointSize', 'Vertex-stage output containing the rasterized point size.'),
  variable('gl_VertexIndex', 'Vulkan vertex-stage input containing the index of the current vertex.'),
  variable('gl_InstanceIndex', 'Vulkan vertex-stage input containing the index of the current instance.'),
  variable('gl_FragCoord', 'Fragment-stage input containing the window-relative fragment coordinates.'),
  variable('gl_FrontFacing', 'Fragment-stage input that is true for front-facing primitives.'),
  variable('gl_PointCoord', 'Fragment-stage input containing point-sprite coordinates.'),
  variable('gl_FragDepth', 'Fragment-stage output used to override the fragment depth value.'),
  variable('gl_GlobalInvocationID', 'Compute-stage input identifying the current global invocation.'),
  variable('gl_LocalInvocationID', 'Compute-stage input identifying the invocation within its work group.'),
  variable('gl_WorkGroupID', 'Compute-stage input identifying the current work group.'),
  variable('gl_LocalInvocationIndex', 'Linear index of the current invocation within its work group.'),
];
