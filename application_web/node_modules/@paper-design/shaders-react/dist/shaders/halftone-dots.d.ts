import { type ShaderComponentProps } from '../shader-mount.js';
import { type HalftoneDotsParams, type ImageShaderPreset } from '@paper-design/shaders';
export interface HalftoneDotsProps extends ShaderComponentProps, HalftoneDotsParams {
}
type HalftoneDotsPreset = ImageShaderPreset<HalftoneDotsParams>;
export declare const defaultPreset: HalftoneDotsPreset;
export declare const ledPreset: HalftoneDotsPreset;
export declare const netPreset: HalftoneDotsPreset;
export declare const roundAndSquarePreset: HalftoneDotsPreset;
export declare const halftoneDotsPresets: HalftoneDotsPreset[];
export declare const HalftoneDots: React.FC<HalftoneDotsProps>;
export {};
