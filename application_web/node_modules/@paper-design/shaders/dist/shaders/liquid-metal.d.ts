import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * Futuristic liquid metal material applied to uploaded logo or abstract shape.
 * Fluid motion imitation applied over user image with animated stripe pattern
 * getting distorted along shape edges.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_resolution (vec2): Canvas resolution in pixels
 * - u_image (sampler2D): Pre-processed source image texture (R = edge gradient, G = opacity)
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colorTint (vec4): Overlay color in RGBA (color burn blending used)
 * - u_repetition (float): Density of pattern stripes (1 to 10)
 * - u_softness (float): Color transition sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
 * - u_shiftRed (float): R-channel dispersion (-1 to 1)
 * - u_shiftBlue (float): B-channel dispersion (-1 to 1)
 * - u_distortion (float): Noise distortion over the stripes pattern (0 to 1)
 * - u_contour (float): Strength of the distortion on the shape edges (0 to 1)
 * - u_angle (float): Direction of pattern animation in degrees (0 to 360)
 * - u_shape (float): Predefined shape when no image provided (0 = none, 1 = circle, 2 = daisy, 3 = diamond, 4 = metaballs)
 * - u_isImage (bool): Whether an image is being used as the effect mask
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_imageUV (vec2): UV coordinates for sampling the source image, with fit, scale, rotation, and offset applied
 * - v_objectUV (vec2): Object box UV coordinates with global sizing (scale, rotation, offsets, etc) applied (used when no image)
 * - v_responsiveUV (vec2): Responsive UV coordinates that adapt to canvas aspect ratio (used for canvas-fill mode)
 * - v_responsiveBoxGivenSize (vec2): TBD
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
export declare const liquidMetalFragmentShader: string;
export declare const POISSON_CONFIG_OPTIMIZED: {
    measurePerformance: boolean;
    workingSize: number;
    iterations: number;
};
export declare function toProcessedLiquidMetal(file: File | string): Promise<{
    imageData: ImageData;
    pngBlob: Blob;
}>;
export interface LiquidMetalUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colorTint: [number, number, number, number];
    u_image: HTMLImageElement | string | undefined;
    u_repetition: number;
    u_shiftRed: number;
    u_shiftBlue: number;
    u_contour: number;
    u_softness: number;
    u_distortion: number;
    u_angle: number;
    u_shape: (typeof LiquidMetalShapes)[LiquidMetalShape];
    u_isImage: boolean;
}
export interface LiquidMetalParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colorTint?: string;
    image?: HTMLImageElement | string | undefined;
    repetition?: number;
    shiftRed?: number;
    shiftBlue?: number;
    contour?: number;
    softness?: number;
    distortion?: number;
    angle?: number;
    shape?: LiquidMetalShape;
}
export declare const LiquidMetalShapes: {
    readonly none: 0;
    readonly circle: 1;
    readonly daisy: 2;
    readonly diamond: 3;
    readonly metaballs: 4;
};
export type LiquidMetalShape = keyof typeof LiquidMetalShapes;
