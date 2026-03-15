import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import type { ShaderSizingParams, ShaderSizingUniforms } from '../shader-sizing.js';
export declare const heatmapMeta: {
    readonly maxColorCount: 10;
};
/**
 * A glowing gradient of colors flowing through an input shape.
 * The effect creates a smoothly animated wave of intensity across the image.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_image (sampler2D): Pre-processed source image texture (R = contour, G = outer blur, B = inner blur)
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colors (vec4[]): Up to 10 heatmap colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_contour (float): Heat intensity near the edges of the input shape (0 to 1)
 * - u_angle (float): Direction of the heatwaves in degrees (0 to 360)
 * - u_noise (float): Grain applied across the entire graphic (0 to 1)
 * - u_innerGlow (float): Size of the heated area inside the input shape (0 to 1)
 * - u_outerGlow (float): Size of the heated area outside the input shape (0 to 1)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_imageUV (vec2): UV coordinates for sampling the source image, with fit, scale, rotation, and offset applied
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
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 *
 */
export declare const heatmapFragmentShader: string;
export declare function toProcessedHeatmap(file: File | string): Promise<{
    blob: Blob;
}>;
export interface HeatmapUniforms extends ShaderSizingUniforms {
    u_image: HTMLImageElement | string;
    u_contour: number;
    u_angle: number;
    u_noise: number;
    u_innerGlow: number;
    u_outerGlow: number;
    u_colorBack: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
}
export interface HeatmapParams extends ShaderSizingParams, ShaderMotionParams {
    image: HTMLImageElement | string;
    contour?: number;
    angle?: number;
    noise?: number;
    innerGlow?: number;
    outerGlow?: number;
    colorBack?: string;
    colors?: string[];
}
