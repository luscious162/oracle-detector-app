import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const simplexNoiseMeta: {
    readonly maxColorCount: 10;
};
/**
 * A multi-color gradient mapped into smooth, animated curves built as a combination of 2 Simplex noises.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_scale (float): Overall zoom level, used for anti-aliasing calculations
 * - u_colors (vec4[]): Up to 10 base colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_stepsPerColor (float): Number of extra colors between base colors, 1 = N colors, 2 = 2×N, etc. (1 to 10)
 * - u_softness (float): Color transition sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
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
export declare const simplexNoiseFragmentShader: string;
export interface SimplexNoiseUniforms extends ShaderSizingUniforms {
    u_colors: vec4[];
    u_colorsCount: number;
    u_stepsPerColor: number;
    u_softness: number;
}
export interface SimplexNoiseParams extends ShaderSizingParams, ShaderMotionParams {
    colors?: string[];
    stepsPerColor?: number;
    softness?: number;
}
