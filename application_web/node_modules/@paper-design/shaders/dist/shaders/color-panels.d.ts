import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const colorPanelsMeta: {
    readonly maxColorCount: 7;
};
/**
 * Pseudo-3D semi-transparent panels rotating around a central axis
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_scale (float): Overall zoom level, used for anti-aliasing calculations
 * - u_colors (vec4[]): Up to 7 RGBA colors used to color the panels
 * - u_colorsCount (float): Number of active colors
 * - u_colorBack (vec4): Background color in RGBA
 * - u_density (float): Angle between every 2 panels (0.25 to 7)
 * - u_angle1 (float): Skew angle applied to all panes (-1 to 1)
 * - u_angle2 (float): Skew angle applied to all panes (-1 to 1)
 * - u_length (float): Panel length relative to total height (0 to 3)
 * - u_edges (bool): Color highlight on the panels edges
 * - u_blur (float): Side blur, 0 for sharp edges (0 to 0.5)
 * - u_fadeIn (float): Transparency near central axis (0 to 1)
 * - u_fadeOut (float): Transparency near viewer (0 to 1)
 * - u_gradient (float): Color mixing within a panel, 0 = solid, 1 = gradient (0 to 1)
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
export declare const colorPanelsFragmentShader: string;
export interface ColorPanelsUniforms extends ShaderSizingUniforms {
    u_colors: vec4[];
    u_colorsCount: number;
    u_colorBack: [number, number, number, number];
    u_angle1: number;
    u_angle2: number;
    u_length: number;
    u_edges: boolean;
    u_blur: number;
    u_fadeIn: number;
    u_fadeOut: number;
    u_density: number;
    u_gradient: number;
}
export interface ColorPanelsParams extends ShaderSizingParams, ShaderMotionParams {
    colors?: string[];
    colorBack?: string;
    angle1?: number;
    angle2?: number;
    length?: number;
    edges?: boolean;
    blur?: number;
    fadeIn?: number;
    fadeOut?: number;
    density?: number;
    gradient?: number;
}
