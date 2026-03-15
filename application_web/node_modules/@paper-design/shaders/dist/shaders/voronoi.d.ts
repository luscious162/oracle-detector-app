import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const voronoiMeta: {
    readonly maxColorCount: 5;
};
/**
 * Anti-aliased animated Voronoi pattern with smooth and customizable edges.
 *
 * Double-pass Voronoi pattern cell edges.
 * Original algorithm: https://www.shadertoy.com/view/ldl3W8
 *
 * Note: gaps can't be removed completely due to natural artifacts of Voronoi cells borders
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_scale (float): Overall zoom level, used for anti-aliasing calculations
 * - u_colors (vec4[]): Up to 5 base cell colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_stepsPerColor (float): Number of extra colors between base colors, 1 = N colors, 2 = 2×N, etc. (1 to 3)
 * - u_colorGlow (vec4): Color tint for radial inner shadow inside cells in RGBA, effective with glow > 0
 * - u_colorGap (vec4): Color used for cell borders/gaps in RGBA
 * - u_distortion (float): Strength of noise-driven displacement of cell centers (0 to 0.5)
 * - u_gap (float): Width of the border/gap between cells (0 to 0.1)
 * - u_glow (float): Strength of the radial inner shadow inside cells (0 to 1)
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
export declare const voronoiFragmentShader: string;
export interface VoronoiUniforms extends ShaderSizingUniforms {
    u_colors: vec4[];
    u_colorsCount: number;
    u_stepsPerColor: number;
    u_colorGap: [number, number, number, number];
    u_colorGlow: [number, number, number, number];
    u_distortion: number;
    u_gap: number;
    u_glow: number;
    u_noiseTexture?: HTMLImageElement;
}
export interface VoronoiParams extends ShaderSizingParams, ShaderMotionParams {
    colors?: string[];
    stepsPerColor?: number;
    colorGap?: string;
    colorGlow?: string;
    distortion?: number;
    gap?: number;
    glow?: number;
}
