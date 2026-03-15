import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const dotOrbitMeta: {
    readonly maxColorCount: 10;
};
/**
 * Animated multi-color dots pattern with each dot orbiting around its cell center.
 * Supports up to 10 colors and various shape and motion controls.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colors (vec4[]): Up to 10 base colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_stepsPerColor (float): Number of extra colors between base colors, 1 = N colors, 2 = 2×N, etc. (1 to 4)
 * - u_size (float): Dot radius relative to cell size (0 to 1)
 * - u_sizeRange (float): Random variation in shape size, 0 = uniform, higher = random up to base size (0 to 1)
 * - u_spreading (float): Maximum orbit distance around cell center (0 to 1)
 * - u_noiseTexture (sampler2D): Pre-computed randomizer source texture
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_patternUV (vec2): UV coordinates in pixels (scaled by 0.01 for precision), with rotation and offset applied
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
export declare const dotOrbitFragmentShader: string;
export interface DotOrbitUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
    u_size: number;
    u_sizeRange: number;
    u_spreading: number;
    u_stepsPerColor: number;
    u_noiseTexture?: HTMLImageElement;
}
export interface DotOrbitParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colors?: string[];
    size?: number;
    sizeRange?: number;
    spreading?: number;
    stepsPerColor?: number;
}
