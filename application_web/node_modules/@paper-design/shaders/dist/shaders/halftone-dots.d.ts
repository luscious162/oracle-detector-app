import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * A halftone-dot image filter featuring customizable grids, color palettes, and dot styles.
 *
 * Fragment shader uniforms:
 * - u_rotation (float): Overall rotation angle of the graphics in degrees (0 to 360)
 * - u_time (float): Animation time
 * - u_image (sampler2D): Source image texture
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorFront (vec4): Foreground color in RGBA
 * - u_colorBack (vec4): Background color in RGBA
 * - u_originalColors (bool): Use sampled image's original colors instead of colorFront
 * - u_type (float): Dot style (0 = classic, 1 = gooey, 2 = holes, 3 = soft)
 * - u_inverted (bool): Inverts the image luminance, doesn't affect the color scheme; not effective at zero contrast
 * - u_grid (float): Grid type (0 = square, 1 = hex)
 * - u_size (float): Grid size relative to the image box (0 to 1)
 * - u_radius (float): Maximum dot size relative to grid cell (0 to 2)
 * - u_contrast (float): Contrast applied to the sampled image (0 to 1)
 * - u_grainMixer (float): Strength of grain distortion applied to shape edges (0 to 1)
 * - u_grainOverlay (float): Post-processing black/white grain overlay (0 to 1)
 * - u_grainSize (float): Scale applied to both grain distortion and grain overlay (0 to 1)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_imageUV (vec2): Image UV coordinates with global sizing (rotation, scale, offset, etc) applied
 *
 * Vertex shader uniforms:
 * - u_resolution (vec2): Canvas resolution in pixels
 * - u_pixelRatio (float): Device pixel ratio
 * - u_originX (float): Reference point for positioning world width in the canvas (0 to 1)
 * - u_originY (float): Reference point for positioning world height in the canvas (0 to 1)
 * - u_fit (float): How to fit the rendered shader into the canvas dimensions (0 = none, 1 = contain, 2 = cover)
 * - u_scale (float): Overall zoom level of the graphics (0.01 to 4)
 * - u_rotation (float): Overall rotation angle of the graphics in degrees (0 to 360)
 * - u_offsetX (float): Horizontal offset of the graphics center (-1 to 1)
 * - u_offsetY (float): Vertical offset of the graphics center (-1 to 1)
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 *
 */
export declare const halftoneDotsFragmentShader: string;
export interface HalftoneDotsUniforms extends ShaderSizingUniforms {
    u_image: HTMLImageElement | string | undefined;
    u_colorFront: [number, number, number, number];
    u_colorBack: [number, number, number, number];
    u_size: number;
    u_grid: (typeof HalftoneDotsGrids)[HalftoneDotsGrid];
    u_radius: number;
    u_contrast: number;
    u_originalColors: boolean;
    u_inverted: boolean;
    u_grainMixer: number;
    u_grainOverlay: number;
    u_grainSize: number;
    u_type: (typeof HalftoneDotsTypes)[HalftoneDotsType];
}
export interface HalftoneDotsParams extends ShaderSizingParams, ShaderMotionParams {
    image?: HTMLImageElement | string;
    colorFront?: string;
    colorBack?: string;
    size?: number;
    grid?: HalftoneDotsGrid;
    radius?: number;
    contrast?: number;
    originalColors?: boolean;
    inverted?: boolean;
    grainMixer?: number;
    grainOverlay?: number;
    grainSize?: number;
    type?: HalftoneDotsType;
}
export declare const HalftoneDotsTypes: {
    readonly classic: 0;
    readonly gooey: 1;
    readonly holes: 2;
    readonly soft: 3;
};
export type HalftoneDotsType = keyof typeof HalftoneDotsTypes;
export declare const HalftoneDotsGrids: {
    readonly square: 0;
    readonly hex: 1;
};
export type HalftoneDotsGrid = keyof typeof HalftoneDotsGrids;
