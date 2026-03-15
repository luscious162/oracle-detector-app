import { type ShaderComponentProps } from '../shader-mount.js';
import { type GrainGradientParams, type ShaderPreset } from '@paper-design/shaders';
export interface GrainGradientProps extends ShaderComponentProps, GrainGradientParams {
}
type GrainGradientPreset = ShaderPreset<GrainGradientParams>;
export declare const defaultPreset: GrainGradientPreset;
export declare const wavePreset: GrainGradientPreset;
export declare const dotsPreset: GrainGradientPreset;
export declare const truchetPreset: GrainGradientPreset;
export declare const ripplePreset: GrainGradientPreset;
export declare const blobPreset: GrainGradientPreset;
export declare const grainGradientPresets: GrainGradientPreset[];
export declare const GrainGradient: React.FC<GrainGradientProps>;
export {};
