import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * A static texture built from multiple noise layers, usable for realistic paper and cardboard surfaces.
 * Can be used as an image filter or as a standalone texture.
 *
 * Fragment shader uniforms:
 * - u_resolution (vec2): Canvas resolution in pixels
 * - u_pixelRatio (float): Device pixel ratio
 * - u_image (sampler2D): Optional source image texture
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorFront (vec4): Foreground color in RGBA
 * - u_colorBack (vec4): Background color in RGBA
 * - u_contrast (float): Blending behavior, sharper vs smoother color transitions (0 to 1)
 * - u_roughness (float): Pixel noise, related to canvas and not scalable (0 to 1)
 * - u_fiber (float): Curly-shaped noise intensity (0 to 1)
 * - u_fiberSize (float): Curly-shaped noise scale (0 to 1)
 * - u_crumples (float): Cell-based crumple pattern intensity (0 to 1)
 * - u_crumpleSize (float): Cell-based crumple pattern scale (0 to 1)
 * - u_folds (float): Depth of the folds (0 to 1)
 * - u_foldCount (float): Number of folds (1 to 15)
 * - u_fade (float): Big-scale noise mask applied to the pattern (0 to 1)
 * - u_drops (float): Visibility of speckle pattern (0 to 1)
 * - u_seed (float): Seed applied to folds, crumples and dots (0 to 1000)
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
export declare const paperTextureFragmentShader: string;
export interface PaperTextureUniforms extends ShaderSizingUniforms {
    u_image: HTMLImageElement | string | undefined;
    u_noiseTexture?: HTMLImageElement;
    u_colorFront: [number, number, number, number];
    u_colorBack: [number, number, number, number];
    u_contrast: number;
    u_roughness: number;
    u_fiber: number;
    u_fiberSize: number;
    u_crumples: number;
    u_foldCount: number;
    u_folds: number;
    u_fade: number;
    u_crumpleSize: number;
    u_drops: number;
    u_seed: number;
}
export interface PaperTextureParams extends ShaderSizingParams, ShaderMotionParams {
    image?: HTMLImageElement | string;
    colorFront?: string;
    colorBack?: string;
    contrast?: number;
    roughness?: number;
    fiber?: number;
    fiberSize?: number;
    crumples?: number;
    foldCount?: number;
    folds?: number;
    fade?: number;
    crumpleSize?: number;
    drops?: number;
    seed?: number;
}
