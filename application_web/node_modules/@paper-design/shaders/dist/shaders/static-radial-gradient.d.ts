import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const staticRadialGradientMeta: {
    readonly maxColorCount: 10;
};
/**
 * Radial gradient with up to 10 blended colors, featuring advanced mixing modes, focal point controls,
 * shape distortion, and grain effects.
 *
 * Fragment shader uniforms:
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colors (vec4[]): Up to 10 gradient colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_radius (float): Size of the shape (0 to 3)
 * - u_focalDistance (float): Distance of the focal point from center (0 to 3)
 * - u_focalAngle (float): Angle of the focal point in degrees, effective with focalDistance > 0 (0 to 360)
 * - u_falloff (float): Gradient decay, 0 = linear gradient (-1 to 1)
 * - u_mixing (float): Blending behavior, 0 = hard stripes, 1 = smooth gradient (0 to 1)
 * - u_distortion (float): Strength of radial distortion (0 to 1)
 * - u_distortionShift (float): Radial distortion offset, effective with distortion > 0 (-1 to 1)
 * - u_distortionFreq (float): Radial distortion frequency, effective with distortion > 0 (0 to 20)
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
export declare const staticRadialGradientFragmentShader: string;
export interface StaticRadialGradientUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
    u_radius: number;
    u_focalDistance: number;
    u_focalAngle: number;
    u_falloff: number;
    u_mixing: number;
    u_distortion: number;
    u_distortionShift: number;
    u_distortionFreq: number;
    u_grainMixer: number;
    u_grainOverlay: number;
}
export interface StaticRadialGradientParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colors?: string[];
    radius?: number;
    focalDistance?: number;
    focalAngle?: number;
    falloff?: number;
    mixing?: number;
    distortion?: number;
    distortionShift?: number;
    distortionFreq?: number;
    grainMixer?: number;
    grainOverlay?: number;
}
