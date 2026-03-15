import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * A glowing, web-like structure of fluid lines and soft intersections.
 * Great for creating atmospheric, organic-yet-futuristic visuals.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_resolution (vec2): Canvas resolution in pixels
 * - u_pixelRatio (float): Device pixel ratio
 * - u_colorFront (vec4): Graphics highlight color in RGBA
 * - u_colorMid (vec4): Graphics main color in RGBA
 * - u_colorBack (vec4): Background color in RGBA
 * - u_brightness (float): Luminosity of the crossing points (0 to 1)
 * - u_contrast (float): Sharpness of the bright-dark transition (0 to 1)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_patternUV (vec2): UV coordinates for pattern with global sizing (rotation, scale, offset, etc) applied
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
 * Original algorithm: https://x.com/zozuar/status/1625182758745128981/
 */
export declare const neuroNoiseFragmentShader: string;
export interface NeuroNoiseUniforms extends ShaderSizingUniforms {
    u_colorFront: [number, number, number, number];
    u_colorMid: [number, number, number, number];
    u_colorBack: [number, number, number, number];
    u_brightness: number;
    u_contrast: number;
}
export interface NeuroNoiseParams extends ShaderSizingParams, ShaderMotionParams {
    colorFront?: string;
    colorMid?: string;
    colorBack?: string;
    brightness?: number;
    contrast?: number;
}
