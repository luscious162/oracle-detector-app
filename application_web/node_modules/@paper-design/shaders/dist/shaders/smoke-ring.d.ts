import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const smokeRingMeta: {
    readonly maxColorCount: 10;
    readonly maxNoiseIterations: 8;
};
/**
 * Radial multi-colored gradient shaped with layered noise for a natural, smoky aesthetic.
 *
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colors (vec4[]): Up to 10 gradient colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_thickness (float): Thickness of the ring shape (0.01 to 1)
 * - u_radius (float): Radius of the ring shape (0 to 1)
 * - u_innerShape (float): Ring inner fill amount (0 to 4)
 * - u_noiseIterations (float): Number of noise layers, more layers gives more details (1 to 8)
 * - u_noiseScale (float): Noise frequency (0.01 to 5)
 * - u_noiseTexture (sampler2D): Pre-computed randomizer source texture
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
export declare const smokeRingFragmentShader: string;
export interface SmokeRingUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
    u_noiseScale: number;
    u_thickness: number;
    u_radius: number;
    u_innerShape: number;
    u_noiseIterations: number;
    u_noiseTexture?: HTMLImageElement;
}
export interface SmokeRingParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colors?: string[];
    noiseScale?: number;
    thickness?: number;
    radius?: number;
    innerShape?: number;
    noiseIterations?: number;
}
