import type { vec4 } from '../types.js';
import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
export declare const pulsingBorderMeta: {
    readonly maxColorCount: 5;
    readonly maxSpots: 4;
};
/**
 * Luminous trails of color merging into a glowing gradient contour.
 *
 * Fragment shader uniforms:
 * - u_time (float): Animation time
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colors (vec4[]): Up to 5 spot colors in RGBA
 * - u_colorsCount (float): Number of active colors
 * - u_roundness (float): Border radius (0 to 1)
 * - u_thickness (float): Border base width (0 to 1)
 * - u_softness (float): Border edge sharpness, 0 = hard edge, 1 = smooth gradient (0 to 1)
 * - u_marginLeft (float): Distance from the left edge to the effect (0 to 1)
 * - u_marginRight (float): Distance from the right edge to the effect (0 to 1)
 * - u_marginTop (float): Distance from the top edge to the effect (0 to 1)
 * - u_marginBottom (float): Distance from the bottom edge to the effect (0 to 1)
 * - u_aspectRatio (float): Aspect ratio mode (0 = auto, 1 = square)
 * - u_intensity (float): Thickness of individual color spots (0 to 1)
 * - u_bloom (float): Power of glow, 0 = normal blending, 1 = additive blending (0 to 1)
 * - u_spots (float): Number of spots added for each color (1 to 20)
 * - u_spotSize (float): Angular size of spots (0 to 1)
 * - u_pulse (float): Optional pulsing animation intensity (0 to 1)
 * - u_smoke (float): Optional noisy shape extending the border (0 to 1)
 * - u_smokeSize (float): Size of the smoke effect (0 to 1)
 * - u_noiseTexture (sampler2D): Pre-computed randomizer source texture
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_responsiveUV (vec2): Responsive UV coordinates that adapt to canvas aspect ratio
 * - v_responsiveBoxGivenSize (vec2): Given size of the responsive bounding box
 * - v_patternUV (vec2): UV coordinates for pattern with global sizing (rotation, scale, offset, etc) applied (used for smoke calculation)
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
export declare const pulsingBorderFragmentShader: string;
export interface PulsingBorderUniforms extends ShaderSizingUniforms {
    u_colorBack: [number, number, number, number];
    u_colors: vec4[];
    u_colorsCount: number;
    u_roundness: number;
    u_thickness: number;
    u_marginLeft: number;
    u_marginRight: number;
    u_marginTop: number;
    u_marginBottom: number;
    u_aspectRatio: (typeof PulsingBorderAspectRatios)[PulsingBorderAspectRatio];
    u_softness: number;
    u_intensity: number;
    u_bloom: number;
    u_spots: number;
    u_spotSize: number;
    u_pulse: number;
    u_smoke: number;
    u_smokeSize: number;
    u_noiseTexture?: HTMLImageElement;
}
export interface PulsingBorderParams extends ShaderSizingParams, ShaderMotionParams {
    colorBack?: string;
    colors?: string[];
    roundness?: number;
    thickness?: number;
    margin?: number;
    marginLeft?: number;
    marginRight?: number;
    marginTop?: number;
    marginBottom?: number;
    aspectRatio?: PulsingBorderAspectRatio;
    softness?: number;
    intensity?: number;
    bloom?: number;
    spots?: number;
    spotSize?: number;
    pulse?: number;
    smoke?: number;
    smokeSize?: number;
}
export declare const PulsingBorderAspectRatios: {
    readonly auto: 0;
    readonly square: 1;
};
export type PulsingBorderAspectRatio = keyof typeof PulsingBorderAspectRatios;
