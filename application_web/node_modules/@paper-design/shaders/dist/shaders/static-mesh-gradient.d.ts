import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const staticMeshGradientMeta: {
    readonly maxColorCount: 10;
};
/**
 * Multi-point mesh gradient with up to 10 color spots, enhanced by two-direction warping,
 * adjustable blend sharpness, and grain controls.
 *
 * Fragment shader uniforms:
 * - u_colors (vec4[]): Up to 10 gradient colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_positions (float): Color spots placement seed (0 to 100)
 * - u_waveX (float): Strength of sine wave distortion along X axis (0 to 1)
 * - u_waveXShift (float): Phase offset applied to the X-axis wave (0 to 1)
 * - u_waveY (float): Strength of sine wave distortion along Y axis (0 to 1)
 * - u_waveYShift (float): Phase offset applied to the Y-axis wave (0 to 1)
 * - u_mixing (float): Blending behavior, 0 = hard stripes, 0.5 = smooth, 1 = gradual blend (0 to 1)
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
export declare const staticMeshGradientFragmentShader: string;
export interface StaticMeshGradientUniforms extends ShaderSizingUniforms {
    u_colors: vec4[];
    u_colorsCount: number;
    u_positions: number;
    u_waveX: number;
    u_waveXShift: number;
    u_waveY: number;
    u_waveYShift: number;
    u_mixing: number;
    u_grainMixer: number;
    u_grainOverlay: number;
}
export interface StaticMeshGradientParams extends ShaderSizingParams, ShaderMotionParams {
    colors?: string[];
    positions?: number;
    waveX?: number;
    waveXShift?: number;
    waveY?: number;
    waveYShift?: number;
    mixing?: number;
    grainMixer?: number;
    grainOverlay?: number;
}
