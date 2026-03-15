import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const warpMeta: {
    readonly maxColorCount: 10;
};
/**
 * Animated color fields warped by noise and swirls, applied over base patterns
 * (checks, stripes, or split edge). Blends up to 10 colors with adjustable distribution,
 * softness, distortion, and swirl. Great for fluid, smoky, or marbled effects.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_scale (float): Overall zoom level, used for anti-aliasing calculations
 * - u_colors (vec4[]): Up to 10 gradient colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_proportion (float): Blend point between colors, 0.5 = equal distribution (0 to 1)
 * - u_softness (float): Color transition sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
 * - u_shape (float): Base pattern type (0 = checks, 1 = stripes, 2 = edge)
 * - u_shapeScale (float): Zoom level of the base pattern (0 to 1)
 * - u_distortion (float): Strength of noise-based distortion (0 to 1)
 * - u_swirl (float): Strength of the swirl distortion (0 to 1)
 * - u_swirlIterations (float): Number of layered swirl passes, effective with swirl > 0 (0 to 20)
 * - u_noiseTexture (sampler2D): Pre-computed randomizer source texture
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
export declare const warpFragmentShader: string;
export interface WarpUniforms extends ShaderSizingUniforms {
    u_colors: vec4[];
    u_colorsCount: number;
    u_proportion: number;
    u_softness: number;
    u_shape: (typeof WarpPatterns)[WarpPattern];
    u_shapeScale: number;
    u_distortion: number;
    u_swirl: number;
    u_swirlIterations: number;
    u_noiseTexture?: HTMLImageElement;
}
export interface WarpParams extends ShaderSizingParams, ShaderMotionParams {
    colors?: string[];
    rotation?: number;
    proportion?: number;
    softness?: number;
    shape?: WarpPattern;
    shapeScale?: number;
    distortion?: number;
    swirl?: number;
    swirlIterations?: number;
}
export declare const WarpPatterns: {
    readonly checks: 0;
    readonly stripes: 1;
    readonly edge: 2;
};
export type WarpPattern = keyof typeof WarpPatterns;
