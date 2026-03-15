import { type ShaderComponentProps } from '../shader-mount.js';
import { type ImageDitheringParams, type ImageShaderPreset } from '@paper-design/shaders';
export interface ImageDitheringProps extends ShaderComponentProps, ImageDitheringParams {
    /** @deprecated use `size` instead */
    pxSize?: number;
}
type ImageDitheringPreset = ImageShaderPreset<ImageDitheringParams>;
export declare const defaultPreset: ImageDitheringPreset;
export declare const retroPreset: ImageDitheringPreset;
export declare const noisePreset: ImageDitheringPreset;
export declare const naturalPreset: ImageDitheringPreset;
export declare const imageDitheringPresets: ImageDitheringPreset[];
export declare const ImageDithering: React.FC<ImageDitheringProps>;
export {};
