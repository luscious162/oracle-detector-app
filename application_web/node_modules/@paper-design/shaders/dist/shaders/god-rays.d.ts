import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const godRaysMeta: {
    readonly maxColorCount: 5;
};
/**
 * Animated rays of light radiating from the center, blended with up to 5 colors.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colorBloom (vec4): Color overlay blended with the rays in RGBA
 * - u_colors (vec4[]): Up to 5 ray colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_bloom (float): Strength of the bloom/overlay effect, 0 = alpha blend, 1 = additive blend (0 to 1)
 * - u_intensity (float): Visibility/strength of the rays (0 to 1)
 * - u_density (float): The number of rays (0 to 1)
 * - u_spotty (float): The length of the rays, higher = more spots/shorter rays (0 to 1)
 * - u_midSize (float): Size of the circular glow shape in the center (0 to 1)
 * - u_midIntensity (float): Brightness/intensity of the central glow (0 to 1)
 * - u_noiseTexture (sampler2D): Pre-computed randomizer source texture
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_objectUV (vec2): Object box UV coordinates with global sizing (scale, rotation, offsets, etc) applied
 *
 * The rays are adjustable by size, density, brightness and center glow.
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
export declare const godRaysFragmentShader: string;
export interface GodRaysUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colorBloom: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
    u_spotty: number;
    u_midSize: number;
    u_midIntensity: number;
    u_density: number;
    u_intensity: number;
    u_bloom: number;
    u_noiseTexture?: HTMLImageElement;
}
export interface GodRaysParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colorBloom?: string;
    colors?: string[];
    spotty?: number;
    midSize?: number;
    midIntensity?: number;
    density?: number;
    intensity?: number;
    bloom?: number;
}
