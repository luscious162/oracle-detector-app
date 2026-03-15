import type { ShaderMotionParams } from '../shader-mount.js';
import { type ShaderSizingParams, type ShaderSizingUniforms } from '../shader-sizing.js';
/**
 * Fluted glass image filter that transforms an image into streaked, ribbed distortions,
 * giving a mix of clarity and obscurity.
 *
 * Fragment shader uniforms:
 * - u_resolution (vec2): Canvas resolution in pixels
 * - u_pixelRatio (float): Device pixel ratio
 * - u_rotation (float): Overall rotation angle of the graphics in degrees (0 to 360)
 * - u_image (sampler2D): Source image texture
 * - u_imageAspectRatio (float): Aspect ratio of the source image
 * - u_colorBack (vec4): Background color in RGBA
 * - u_colorShadow (vec4): Shadows color in RGBA
 * - u_colorHighlight (vec4): Highlights color in RGBA
 * - u_shadows (float): Color gradient added over image and background, following distortion shape (0 to 1)
 * - u_highlights (float): Thin strokes along distortion shape, useful for antialiasing on small grid (0 to 1)
 * - u_size (float): Size of the distortion shape grid (0 to 1)
 * - u_shape (float): Grid shape (1 = lines, 2 = linesIrregular, 3 = wave, 4 = zigzag, 5 = pattern)
 * - u_angle (float): Direction of the grid relative to the image in degrees (0 to 180)
 * - u_distortionShape (float): Shape of distortion (1 = prism, 2 = lens, 3 = contour, 4 = cascade, 5 = flat)
 * - u_distortion (float): Power of distortion applied within each stripe (0 to 1)
 * - u_shift (float): Texture shift in direction opposite to the grid (-1 to 1)
 * - u_stretch (float): Extra distortion along the grid lines (0 to 1)
 * - u_blur (float): One-directional blur over the image and extra blur around edges (0 to 1)
 * - u_edges (float): Glass distortion and softness on the image edges (0 to 1)
 * - u_marginLeft (float): Distance from the left edge to the effect (0 to 1)
 * - u_marginRight (float): Distance from the right edge to the effect (0 to 1)
 * - u_marginTop (float): Distance from the top edge to the effect (0 to 1)
 * - u_marginBottom (float): Distance from the bottom edge to the effect (0 to 1)
 * - u_grainMixer (float): Strength of grain distortion applied to shape edges (0 to 1)
 * - u_grainOverlay (float): Post-processing black/white grain overlay (0 to 1)
 *
 * Vertex shader outputs (used in fragment shader):
 * - v_imageUV (vec2): Image UV coordinates with global sizing (rotation, scale, offset, etc) applied
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
export declare const flutedGlassFragmentShader: string;
export interface FlutedGlassUniforms extends ShaderSizingUniforms {
    u_image: HTMLImageElement | string | undefined;
    u_colorBack: [number, number, number, number];
    u_colorShadow: [number, number, number, number];
    u_colorHighlight: [number, number, number, number];
    u_shadows: number;
    u_size: number;
    u_angle: number;
    u_distortion: number;
    u_shift: number;
    u_blur: number;
    u_edges: number;
    u_marginLeft: number;
    u_marginRight: number;
    u_marginTop: number;
    u_marginBottom: number;
    u_stretch: number;
    u_distortionShape: (typeof GlassDistortionShapes)[GlassDistortionShape];
    u_highlights: number;
    u_shape: (typeof GlassGridShapes)[GlassGridShape];
    u_grainMixer: number;
    u_grainOverlay: number;
    u_noiseTexture?: HTMLImageElement;
}
export interface FlutedGlassParams extends ShaderSizingParams, ShaderMotionParams {
    image?: HTMLImageElement | string;
    colorBack?: string;
    colorShadow?: string;
    colorHighlight?: string;
    shadows?: number;
    size?: number;
    angle?: number;
    distortion?: number;
    shift?: number;
    blur?: number;
    edges?: number;
    margin?: number;
    marginLeft?: number;
    marginRight?: number;
    marginTop?: number;
    marginBottom?: number;
    stretch?: number;
    distortionShape?: GlassDistortionShape;
    highlights?: number;
    shape?: GlassGridShape;
    grainMixer?: number;
    grainOverlay?: number;
}
export declare const GlassGridShapes: {
    readonly lines: 1;
    readonly linesIrregular: 2;
    readonly wave: 3;
    readonly zigzag: 4;
    readonly pattern: 5;
};
export declare const GlassDistortionShapes: {
    readonly prism: 1;
    readonly lens: 2;
    readonly contour: 3;
    readonly cascade: 4;
    readonly flat: 5;
};
export type GlassDistortionShape = keyof typeof GlassDistortionShapes;
export type GlassGridShape = keyof typeof GlassGridShapes;
