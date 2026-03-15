import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const meshGradientMeta: {
    readonly maxColorCount: 10;
};
/**
 * A flowing composition of color spots, moving along distinct trajectories
 * and transformed by organic distortion.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colors (vec4[]): Up to 10 color spots in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_distortion (float): Power of organic noise distortion (0 to 1)
 * - u_swirl (float): Power of vortex distortion (0 to 1)
 * - u_grainMixer (float): Strength of grain distortion applied to shape edges (0 to 1)
 * - u_grainOverlay (float): Post-processing black/white grain overlay (0 to 1)
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
export declare const meshGradientFragmentShader: string;
export interface MeshGradientUniforms extends ShaderSizingUniforms {
    u_colors: vec4[];
    u_colorsCount: number;
    u_distortion: number;
    u_swirl: number;
    u_grainMixer: number;
    u_grainOverlay: number;
}
export interface MeshGradientParams extends ShaderSizingParams, ShaderMotionParams {
    colors?: string[];
    distortion?: number;
    swirl?: number;
    grainMixer?: number;
    grainOverlay?: number;
}
