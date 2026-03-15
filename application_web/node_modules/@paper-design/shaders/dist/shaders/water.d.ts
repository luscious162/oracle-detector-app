import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * Water-like surface distortion with natural caustic realism. Works as an image filter or standalone animated texture.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_image (sampler2D): Optional source image texture
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colorHighlight (vec4): Highlight color in RGBA
 * - u_highlights (float): Coloring added over image/background following caustic shape (0 to 1)
 * - u_layering (float): Power of 2nd layer of caustic distortion (0 to 1)
 * - u_edges (float): Caustic distortion power on the image edges (0 to 1)
 * - u_waves (float): Additional distortion based on simplex noise, independent from caustic (0 to 1)
 * - u_caustic (float): Power of caustic distortion (0 to 1)
 * - u_size (float): Pattern scale relative to the image (0.01 to 7)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_imageUV (vec2): UV coordinates for sampling the source image, with fit, scale, rotation, and offset applied
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
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 *
 */
export declare const waterFragmentShader: string;
export interface WaterUniforms extends ShaderSizingUniforms {
    u_image: HTMLImageElement | string | undefined;
    u_colorBack: [number, number, number, number];
    u_colorHighlight: [number, number, number, number];
    u_highlights: number;
    u_layering: number;
    u_edges: number;
    u_caustic: number;
    u_waves: number;
    u_size: number;
}
export interface WaterParams extends ShaderSizingParams, ShaderMotionParams {
    image?: HTMLImageElement | string;
    colorBack?: string;
    colorHighlight?: string;
    highlights?: number;
    layering?: number;
    edges?: number;
    caustic?: number;
    waves?: number;
    size?: number;
}
