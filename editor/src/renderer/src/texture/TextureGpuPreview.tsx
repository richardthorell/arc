import { useEffect, useRef } from 'react';

import { evaluateTextureCurve } from './texturePixelProcessing';
import type { TexturePreviewMode } from './texturePreviewProcessing';
import type { TextureChannelSource, TextureSettingsSnapshot } from './textureSettings';

type Channels = { r: boolean; g: boolean; b: boolean; a: boolean };

const vertexSource = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const fragmentSource = `#version 300 es
precision highp float;
precision highp int;
in vec2 uv;
out vec4 color;
uniform sampler2D sourceTexture;
uniform sampler2D curveTexture;
uniform ivec4 mapping;
uniform vec4 inversion;
uniform vec4 levels;
uniform vec4 adjustment;
uniform vec4 tintAndVibrance;
uniform vec4 displayChannels;
uniform int mode;
uniform bool curvesEnabled;
uniform bool normalMap;
uniform bool srgb;
uniform float exposure;

float curve(float x, int row) {
  return texelFetch(curveTexture, ivec2(int(clamp(x, 0.0, 1.0) * 255.0 + 0.5), row), 0).r;
}
float channel(vec4 pixel, int source) {
  if (source == 0) return pixel.r;
  if (source == 1) return pixel.g;
  if (source == 2) return pixel.b;
  if (source == 3) return pixel.a;
  return source == 5 ? 1.0 : 0.0;
}
vec3 toLinear(vec3 v) {
  return mix(v / 12.92, pow((v + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), v));
}
vec3 toSrgb(vec3 v) {
  v = clamp(v, 0.0, 1.0);
  return mix(v * 12.92, 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), v));
}
vec4 process(vec4 raw) {
  vec4 v = vec4(channel(raw, mapping.x), channel(raw, mapping.y), channel(raw, mapping.z), channel(raw, mapping.w));
  v = mix(v, 1.0 - v, inversion);
  if (!normalMap) {
    if (srgb) v.rgb = toLinear(v.rgb);
    v.rgb = clamp((v.rgb - levels.x) / max(0.0001, levels.y - levels.x), 0.0, 1.0);
    if (curvesEnabled) {
      v.r = curve(curve(v.r, 1), 0);
      v.g = curve(curve(v.g, 2), 0);
      v.b = curve(curve(v.b, 3), 0);
    }
    v.rgb = pow(clamp(v.rgb, 0.0, 1.0), vec3(1.0 / adjustment.x));
    v.rgb *= exp2(adjustment.y);
    v.rgb = (v.rgb - 0.5) * adjustment.z + 0.5;
    float luminance = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722));
    v.rgb = vec3(luminance) + (v.rgb - luminance) * adjustment.w;
    float spread = clamp(max(max(v.r, v.g), v.b) - min(min(v.r, v.g), v.b), 0.0, 1.0);
    v.rgb = vec3(luminance) + (v.rgb - luminance) * (1.0 + tintAndVibrance.w * (1.0 - spread));
    v.rgb *= tintAndVibrance.rgb;
    v.rgb = levels.z + clamp(v.rgb, 0.0, 1.0) * (levels.w - levels.z);
    v.rgb = srgb ? toSrgb(v.rgb) : clamp(v.rgb, 0.0, 1.0);
  }
  if (curvesEnabled) v.a = curve(v.a, 4);
  return clamp(v, 0.0, 1.0);
}
void main() {
  vec4 raw = texture(sourceTexture, uv);
  vec4 value = mode == 0 ? raw : process(raw);
  if (mode == 2) value = vec4(min(abs(value.rgb - raw.rgb) * 4.0, 1.0), 1.0);
  bool anyRgb = displayChannels.r + displayChannels.g + displayChannels.b > 0.0;
  vec3 rgb = anyRgb ? value.rgb * displayChannels.rgb : vec3(value.a);
  rgb *= exp2(exposure);
  float checker = mod(floor(gl_FragCoord.x / 8.0) + floor(gl_FragCoord.y / 8.0), 2.0) == 0.0 ? 0.25 : 0.38;
  float alpha = displayChannels.a > 0.0 && anyRgb ? value.a : 1.0;
  color = vec4(mix(vec3(checker), rgb, alpha), 1.0);
}`;

function shader(gl: WebGL2RenderingContext, type: number, source: string) {
  const result = gl.createShader(type);
  if (!result) throw new Error('Could not create texture preview shader');
  gl.shaderSource(result, source);
  gl.compileShader(result);
  if (!gl.getShaderParameter(result, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(result) || 'Shader failed');
  return result;
}

const channelIndex = (source: TextureChannelSource) =>
  ({ red: 0, green: 1, blue: 2, alpha: 3, zero: 4, one: 5 })[source];

/** Browser GPU preview for uncommitted import settings; it never mutates the cooked asset. */
export function TextureGpuPreview({
  dataUrl,
  settings,
  mode,
  channels,
  exposure,
  sampling,
  onPointerMove,
  onPointerLeave,
}: {
  dataUrl: string;
  settings: TextureSettingsSnapshot;
  mode: TexturePreviewMode;
  channels: Channels;
  exposure: number;
  sampling: 'nearest' | 'linear';
  onPointerMove?: React.PointerEventHandler<HTMLCanvasElement>;
  onPointerLeave?: React.PointerEventHandler<HTMLCanvasElement>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gpuRef = useRef<{
    gl: WebGL2RenderingContext;
    program: WebGLProgram;
    source: WebGLTexture;
    curves: WebGLTexture;
  } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas?.getContext('webgl2', { alpha: false, antialias: false, preserveDrawingBuffer: true });
    if (!canvas || !gl) return;
    const program = gl.createProgram();
    const source = gl.createTexture();
    const curves = gl.createTexture();
    if (!program || !source || !curves) return;
    const vertex = shader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragment = shader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(program) || 'Preview link failed');
    gpuRef.current = { gl, program, source, curves };
    return () => {
      gpuRef.current = null;
      gl.deleteTexture(source);
      gl.deleteTexture(curves);
      gl.deleteProgram(program);
    };
  }, []);

  useEffect(() => {
    const gpu = gpuRef.current;
    if (!gpu) return;
    let active = true;
    const image = new Image();
    image.src = dataUrl;
    void image.decode().then(() => {
      if (!active || !canvasRef.current) return;
      const { gl, source } = gpu;
      canvasRef.current.width = image.naturalWidth;
      canvasRef.current.height = image.naturalHeight;
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, source);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      canvasRef.current.dispatchEvent(new Event('texture-preview-ready'));
    });
    return () => {
      active = false;
    };
  }, [dataUrl]);

  useEffect(() => {
    const gpu = gpuRef.current;
    const canvas = canvasRef.current;
    if (!gpu || !canvas) return;
    const render = () => {
      if (!canvas.width || !canvas.height) return;
      const { gl, program, source, curves } = gpu;
      const uniform = (name: string) => gl.getUniformLocation(program, name);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, source);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, sampling === 'nearest' ? gl.NEAREST : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, sampling === 'nearest' ? gl.NEAREST : gl.LINEAR);
      gl.uniform1i(uniform('sourceTexture'), 0);
      const lookup = new Uint8Array(256 * 5 * 4);
      [settings.curveMaster, settings.curveR, settings.curveG, settings.curveB, settings.curveA].forEach(
        (curve, row) => {
          for (let x = 0; x < 256; x += 1) {
            const offset = (row * 256 + x) * 4;
            lookup[offset] = Math.round(evaluateTextureCurve(curve, x / 255) * 255);
            lookup[offset + 3] = 255;
          }
        },
      );
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, curves);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 5, 0, gl.RGBA, gl.UNSIGNED_BYTE, lookup);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.uniform1i(uniform('curveTexture'), 1);
      gl.uniform4i(
        uniform('mapping'),
        channelIndex(settings.channelR),
        channelIndex(settings.channelG),
        channelIndex(settings.channelB),
        channelIndex(settings.channelA),
      );
      gl.uniform4f(
        uniform('inversion'),
        Number(settings.invertR),
        Number(settings.invertG),
        Number(settings.invertB),
        Number(settings.invertA),
      );
      gl.uniform4f(
        uniform('levels'),
        settings.inputBlack,
        settings.inputWhite,
        settings.outputBlack,
        settings.outputWhite,
      );
      gl.uniform4f(uniform('adjustment'), settings.gamma, settings.brightness, settings.contrast, settings.saturation);
      gl.uniform4f(uniform('tintAndVibrance'), settings.tintR, settings.tintG, settings.tintB, settings.vibrance);
      gl.uniform4f(
        uniform('displayChannels'),
        Number(channels.r),
        Number(channels.g),
        Number(channels.b),
        Number(channels.a),
      );
      gl.uniform1i(uniform('mode'), mode === 'source' ? 0 : mode === 'processed' ? 1 : 2);
      gl.uniform1i(uniform('curvesEnabled'), Number(settings.curvesEnabled));
      gl.uniform1i(uniform('normalMap'), Number(settings.semantic === 'normal'));
      gl.uniform1i(uniform('srgb'), Number(settings.colorSpace === 'srgb'));
      gl.uniform1f(uniform('exposure'), exposure);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    canvas.addEventListener('texture-preview-ready', render);
    const frame = window.requestAnimationFrame(render);
    return () => {
      canvas.removeEventListener('texture-preview-ready', render);
      window.cancelAnimationFrame(frame);
    };
  }, [channels.a, channels.b, channels.g, channels.r, exposure, mode, sampling, settings]);

  return (
    <canvas
      aria-label="GPU texture preview"
      className="texture-gpu-preview"
      onPointerLeave={onPointerLeave}
      onPointerMove={onPointerMove}
      ref={canvasRef}
    />
  );
}
