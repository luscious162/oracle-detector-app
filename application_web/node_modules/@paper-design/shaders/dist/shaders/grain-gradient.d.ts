import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const grainGradientMeta: {
    readonly maxColorCount: 7;
};
/**
 * Multi-color gradients with grainy, noise-textured distortion available in 7 animated abstract forms.
 *
 * Note: grains are calculated using gl_FragCoord & u_resolution, meaning grains don't react to scaling and fit
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_resolution (vec2): Canvas resolution in pixels
 * - u_pixelRatio (float): Device pixel ratio
 * - u_originX (float): Reference point for positioning (0 to 1)
 * - u_originY (float): Reference point for positioning (0 to 1)
 * - u_worldWidth (float): Virtual width of the graphic
 * - u_worldHeight (float): Virtual height of the graphic
 * - u_fit (float): Fit mode (0 = none, 1 = contain, 2 = cover)
 * - u_scale (float): Overall zoom level (0.01 to 4)
 * - u_rotation (float): Rotation angle in degrees (0 to 360)
 * - u_offsetX (float): Horizontal offset (-1 to 1)
 * - u_offsetY (float): Vertical offset (-1 to 1)
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colors (vec4[]): Up to 7 gradient colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_softness (float): Color transition sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
 * - u_intensity (float): Distortion between color bands (0 to 1)
 * - u_noise (float): Grainy noise overlay (0 to 1)
 * - u_shape (float): Shape type (1 = wave, 2 = dots, 3 = truchet, 4 = corners, 5 = ripple, 6 = blob, 7 = sphere)
 * - u_noiseTexture (sampler2D): Pre-computed randomizer source texture
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_objectUV (vec2): Object box UV coordinates with global sizing (scale, rotation, offsets, etc) applied (used for shapes 4-7)
 * - v_objectBoxSize (vec2): Size of the object bounding box in pixels
 * - v_patternUV (vec2): UV coordinates for pattern with global sizing (rotation, scale, offset, etc) applied (used for shapes 1-3)
 * - v_patternBoxSize (vec2): Size of the pattern bounding box in pixels
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
export declare const grainGradientFragmentShader: string;
export interface GrainGradientUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
    u_softness: number;
    u_intensity: number;
    u_noise: number;
    u_shape: (typeof GrainGradientShapes)[GrainGradientShape];
    u_noiseTexture?: HTMLImageElement;
}
export interface GrainGradientParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colors?: string[];
    softness?: number;
    intensity?: number;
    noise?: number;
    shape?: GrainGradientShape;
}
export declare const GrainGradientShapes: {
    wave: number;
    dots: number;
    truchet: number;
    corners: number;
    ripple: number;
    blob: number;
    sphere: number;
};
export type GrainGradientShape = keyof typeof GrainGradientShapes;
