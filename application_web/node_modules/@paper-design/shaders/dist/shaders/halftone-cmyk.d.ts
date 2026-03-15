import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * CMYK halftone printing effect applied to images with customizable dot patterns
 * and ink colors for each channel (Cyan, Magenta, Yellow, Black).
 *
 * Fragment shader uniforms:
 * - u_image (sampler2D): Source image texture
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorBack (vec4): Background (paper) color in RGBA
 * - u_colorC (vec4): Cyan ink color in RGBA
 * - u_colorM (vec4): Magenta ink color in RGBA
 * - u_colorY (vec4): Yellow ink color in RGBA
 * - u_colorK (vec4): Black ink color in RGBA
 * - u_size (float): Halftone cell size (0 to 1)
 * - u_minDot (float): Minimum dot thickness (0 to 1)
 * - u_contrast (float): Image contrast adjustment (0 to 2)
 * - u_softness (float): Edge softness of dots (0 to 1)
 * - u_grainSize (float): Size of grain overlay texture (0 to 1)
 * - u_grainMixer (float): Strength of grain affecting dot size (0 to 1)
 * - u_grainOverlay (float): Strength of grain overlay on final output (0 to 1)
 * - u_gridNoise (float): Strength of smooth noise applied to both dot positions and color sampling (0 to 1)
 * - u_floodC (float): Flat cyan dot size adjustment applied uniformly (-1 to 1)
 * - u_floodM (float): Flat magenta dot size adjustment applied uniformly (-1 to 1)
 * - u_floodY (float): Flat yellow dot size adjustment applied uniformly (-1 to 1)
 * - u_floodK (float): Flat black dot size adjustment applied uniformly (-1 to 1)
 * - u_gainC (float): Proportional cyan dot size gain (enhances existing dots, -1 to 1)
 * - u_gainM (float): Proportional magenta dot size gain (enhances existing dots, -1 to 1)
 * - u_gainY (float): Proportional yellow dot size gain (enhances existing dots, -1 to 1)
 * - u_gainK (float): Proportional black dot size gain (enhances existing dots, -1 to 1)
 * - u_type (float): Dot shape style (0 = dots, 1 = ink, 2 = sharp)
 * - u_noiseTexture (sampler2D): Pre-computed randomizer source texture
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_imageUV (vec2): UV coordinates for sampling the source image, with fit, scale, rotation, and offset applied
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
export declare const halftoneCmykFragmentShader: string;
export interface HalftoneCmykUniforms extends ShaderSizingUniforms {
    u_image: HTMLImageElement | string | undefined;
    u_noiseTexture?: HTMLImageElement;
    u_colorBack: [number, number, number, number];
    u_colorC: [number, number, number, number];
    u_colorM: [number, number, number, number];
    u_colorY: [number, number, number, number];
    u_colorK: [number, number, number, number];
    u_size: number;
    u_contrast: number;
    u_softness: number;
    u_grainSize: number;
    u_grainMixer: number;
    u_grainOverlay: number;
    u_gridNoise: number;
    u_floodC: number;
    u_floodM: number;
    u_floodY: number;
    u_floodK: number;
    u_gainC: number;
    u_gainM: number;
    u_gainY: number;
    u_gainK: number;
    u_type: (typeof HalftoneCmykTypes)[HalftoneCmykType];
}
export interface HalftoneCmykParams extends ShaderSizingParams, ShaderMotionParams {
    image?: HTMLImageElement | string;
    colorBack?: string;
    colorC?: string;
    colorM?: string;
    colorY?: string;
    colorK?: string;
    size?: number;
    contrast?: number;
    softness?: number;
    grainSize?: number;
    grainMixer?: number;
    grainOverlay?: number;
    gridNoise?: number;
    floodC?: number;
    floodM?: number;
    floodY?: number;
    floodK?: number;
    gainC?: number;
    gainM?: number;
    gainY?: number;
    gainK?: number;
    type?: HalftoneCmykType;
}
export declare const HalftoneCmykTypes: {
    readonly dots: 0;
    readonly ink: 1;
    readonly sharp: 2;
};
export type HalftoneCmykType = keyof typeof HalftoneCmykTypes;
