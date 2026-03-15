import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * Static grid pattern made of circles, diamonds, squares or triangles.
 *
 * Fragment shader uniforms:
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colorFill (vec4): Shape fill color in RGBA
 * - u_colorStroke (vec4): Shape stroke color in RGBA
 * - u_dotSize (float): Base size of each shape in pixels (1 to 100)
 * - u_gapX (float): Pattern horizontal spacing in pixels (2 to 500)
 * - u_gapY (float): Pattern vertical spacing in pixels (2 to 500)
 * - u_strokeWidth (float): Outline stroke width in pixels (0 to 50)
 * - u_sizeRange (float): Random variation in shape size, 0 = uniform, higher = random up to base size (0 to 1)
 * - u_opacityRange (float): Random variation in shape opacity, 0 = opaque, higher = semi-transparent (0 to 1)
 * - u_shape (float): Shape type (0 = circle, 1 = diamond, 2 = square, 3 = triangle)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_patternUV (vec2): UV coordinates in pixels (scaled by 0.01 for precision), with scale, rotation and offset applied
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
export declare const dotGridFragmentShader: string;
export interface DotGridUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colorFill: [number, number, number, number];
    u_colorStroke: [number, number, number, number];
    u_dotSize: number;
    u_gapX: number;
    u_gapY: number;
    u_strokeWidth: number;
    u_sizeRange: number;
    u_opacityRange: number;
    u_shape: (typeof DotGridShapes)[DotGridShape];
}
export interface DotGridParams extends ShaderSizingParams {
    colorBack?: string;
    colorFill?: string;
    colorStroke?: string;
    size?: number;
    gapX?: number;
    gapY?: number;
    strokeWidth?: number;
    sizeRange?: number;
    opacityRange?: number;
    shape?: DotGridShape;
}
export declare const DotGridShapes: {
    readonly circle: 0;
    readonly diamond: 1;
    readonly square: 2;
    readonly triangle: 3;
};
export type DotGridShape = keyof typeof DotGridShapes;
