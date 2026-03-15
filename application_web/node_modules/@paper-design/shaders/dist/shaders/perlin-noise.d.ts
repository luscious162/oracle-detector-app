import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * Classic animated 3D Perlin noise with exposed controls.
 * Original algorithm: https://www.shadertoy.com/view/NlSGDz
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colorFront (vec4): Foreground color in RGBA
 * - u_colorBack (vec4): Background color in RGBA
 * - u_proportion (float): Blend point between 2 colors, 0.5 = equal distribution (0 to 1)
 * - u_softness (float): Color transition sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
 * - u_octaveCount (float): Perlin noise octaves number, more octaves for more detailed patterns (1 to 8)
 * - u_persistence (float): Roughness, falloff between octaves (0.3 to 1)
 * - u_lacunarity (float): Frequency step, defines how compressed the pattern is (1.5 to 10)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_patternUV (vec2): UV coordinates for pattern with global sizing (rotation, scale, offset, etc) applied
 *
 * Vertex shader uniforms:
 * - u_resolution (vec2): Canvas resolution in pixels
 * - u_pixelRatio (float): Device pixel ratio
 * - u_originX (float): Reference point for positioning world width in the canvas (0 to 1)
 * - u_originY (float): Reference point for positioning world height in the canvas (0 to 1)
 * - u_worldWidth (float): Virtual width of the graphic before it's scaled to fit the canvas
 * - u_worldHeight (float): Virtual height of the graphic before it's scaled to fit the canvas
 * - u_fit (float): How to fit the rendered shader into the canvas dimensions (0 = none, 1 = contain, 2 = cover)
 * - u_scale (float): Overall zoom level of the graphics (0.01 to 4)
 * - u_rotation (float): Overall rotation angle of the graphics in degrees (0 to 360)
 * - u_offsetX (float): Horizontal offset of the graphics center (-1 to 1)
 * - u_offsetY (float): Vertical offset of the graphics center (-1 to 1)
 *
 */
export declare const perlinNoiseFragmentShader: string;
export interface PerlinNoiseUniforms extends ShaderSizingUniforms {
    u_colorFront: [number, number, number, number];
    u_colorBack: [number, number, number, number];
    u_proportion: number;
    u_softness: number;
    u_octaveCount: number;
    u_persistence: number;
    u_lacunarity: number;
}
export interface PerlinNoiseParams extends ShaderSizingParams, ShaderMotionParams {
    colorFront?: string;
    colorBack?: string;
    proportion?: number;
    softness?: number;
    octaveCount?: number;
    persistence?: number;
    lacunarity?: number;
}
