import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * A dithering image filter with support for 4 dithering modes and multiple color palettes
 * (2-color, 3-color, and multicolor options, using either predefined colors or colors sampled
 * from the original image).
 *
 * SIZING NOTE: This shader performs sizing in the fragment shader (not vertex shader) to keep
 * u_pxSize in consistent actual pixels. The pixel grid is computed from gl_FragCoord before any
 * transforms, so scaling/rotating only affects the underlying image.
 * No vertex shader outputs (v_imageUV, v_objectUV, etc.) are used.
 *
 * Fragment shader uniforms:
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
 * - u_image (sampler2D): Source image texture
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorFront (vec4): Foreground color in RGBA
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colorHighlight (vec4): Secondary foreground color in RGBA (set same as colorFront for classic 2-color dithering)
 * - u_originalColors (bool): Use the original colors of the image instead of the color palette
 * - u_inverted (bool): Inverts the image luminance, doesn't affect the color scheme; not effective at zero contrast
 * - u_type (float): Dithering type (1 = random, 2 = 2x2 Bayer, 3 = 4x4 Bayer, 4 = 8x8 Bayer)
 * - u_pxSize (float): Pixel size of dithering grid (0.5 to 20)
 * - u_colorSteps (float): Number of colors to use, applies to both color modes (1 to 7)
 *
 */
export declare const imageDitheringFragmentShader: string;
export interface ImageDitheringUniforms extends ShaderSizingUniforms {
    u_image: HTMLImageElement | string;
    u_colorFront: [number, number, number, number];
    u_colorBack: [number, number, number, number];
    u_colorHighlight: [number, number, number, number];
    u_type: (typeof DitheringTypes)[DitheringType];
    u_pxSize: number;
    u_colorSteps: number;
    u_originalColors: boolean;
    u_inverted: boolean;
}
export interface ImageDitheringParams extends ShaderSizingParams, ShaderMotionParams {
    image: HTMLImageElement | string;
    colorFront?: string;
    colorBack?: string;
    colorHighlight?: string;
    type?: DitheringType;
    size?: number;
    colorSteps?: number;
    originalColors?: boolean;
    inverted?: boolean;
}
export declare const DitheringTypes: {
    readonly random: 1;
    readonly '2x2': 2;
    readonly '4x4': 3;
    readonly '8x8': 4;
};
export type DitheringType = keyof typeof DitheringTypes;
