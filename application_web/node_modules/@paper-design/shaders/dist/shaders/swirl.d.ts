import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const swirlMeta: {
    readonly maxColorCount: 10;
};
/**
 * Animated bands of color twisting and bending, producing spirals, arcs, and flowing circular patterns.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colors (vec4[]): Up to 10 stripe colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_bandCount (float): Number of color bands, 0 = concentric ripples (0 to 15)
 * - u_twist (float): Vortex power, 0 = straight sectoral shapes (0 to 1)
 * - u_center (float): How far from the center the swirl colors begin to appear (0 to 1)
 * - u_proportion (float): Blend point between colors, 0.5 = equal distribution (0 to 1)
 * - u_softness (float): Color transition sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
 * - u_noise (float): Strength of noise distortion, no effect with noiseFrequency = 0 (0 to 1)
 * - u_noiseFrequency (float): Noise frequency, no effect with noise = 0 (0 to 1)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_objectUV (vec2): Object box UV coordinates with global sizing (scale, rotation, offsets, etc) applied
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
export declare const swirlFragmentShader: string;
export interface SwirlUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
    u_bandCount: number;
    u_twist: number;
    u_center: number;
    u_proportion: number;
    u_softness: number;
    u_noiseFrequency: number;
    u_noise: number;
}
export interface SwirlParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colors?: string[];
    bandCount?: number;
    twist?: number;
    center?: number;
    proportion?: number;
    softness?: number;
    noiseFrequency?: number;
    noise?: number;
}
