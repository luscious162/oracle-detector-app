import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * A single-colored animated spiral that morphs across a wide range of shapes -
 * from crisp, thin-lined geometry to flowing whirlpool forms and wavy, abstract rings.
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
 * Vertex shader outputs (used in fragment shader):
 * - v_patternUV (vec2): UV coordinates in pixels (scaled by 0.01 for precision), with rotation and offset applied
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colorFront (vec4): Foreground (ink) color in RGBA
 * - u_density (float): Spacing falloff simulating perspective, 0 = flat spiral (0 to 1)
 * - u_distortion (float): Power of shape distortion applied along the spiral (0 to 1)
 * - u_strokeWidth (float): Thickness of spiral curve (0 to 1)
 * - u_strokeTaper (float): How much stroke loses width away from center, 0 = full visibility (0 to 1)
 * - u_strokeCap (float): Extra stroke width at the center, no effect with strokeWidth = 0.5 (0 to 1)
 * - u_noise (float): Noise distortion applied over the canvas, no effect with noiseFrequency = 0 (0 to 1)
 * - u_noiseFrequency (float): Noise frequency, no effect with noise = 0 (0 to 1)
 * - u_softness (float): Color transition sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
 *
 */
export declare const spiralFragmentShader: string;
export interface SpiralUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colorFront: [number, number, number, number];
    u_density: number;
    u_distortion: number;
    u_strokeWidth: number;
    u_strokeTaper: number;
    u_strokeCap: number;
    u_noise: number;
    u_noiseFrequency: number;
    u_softness: number;
}
export interface SpiralParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colorFront?: string;
    density?: number;
    distortion?: number;
    strokeWidth?: number;
    strokeTaper?: number;
    strokeCap?: number;
    noise?: number;
    noiseFrequency?: number;
    softness?: number;
}
