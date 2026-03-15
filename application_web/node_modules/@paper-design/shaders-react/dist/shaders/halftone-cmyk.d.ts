import { type ShaderComponentProps } from '../shader-mount.js';
import { type HalftoneCmykParams, type ImageShaderPreset } from '@paper-design/shaders';
export interface HalftoneCmykProps extends ShaderComponentProps, HalftoneCmykParams {
}
type HalftoneCmykPreset = ImageShaderPreset<HalftoneCmykParams>;
export declare const defaultPreset: HalftoneCmykPreset;
export declare const dropsPreset: HalftoneCmykPreset;
export declare const newspaper: HalftoneCmykPreset;
export declare const vintagePreset: HalftoneCmykPreset;
export declare const halftoneCmykPresets: HalftoneCmykPreset[];
export declare const HalftoneCmyk: React.FC<HalftoneCmykProps>;
export {};
