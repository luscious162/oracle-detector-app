import { type ShaderComponentProps } from '../shader-mount.js';
import { type PerlinNoiseParams, type ShaderPreset } from '@paper-design/shaders';
export interface PerlinNoiseProps extends ShaderComponentProps, PerlinNoiseParams {
}
type PerlinNoisePreset = ShaderPreset<PerlinNoiseParams>;
export declare const defaultPreset: PerlinNoisePreset;
export declare const nintendoWaterPreset: PerlinNoisePreset;
export declare const mossPreset: PerlinNoisePreset;
export declare const wormsPreset: PerlinNoisePreset;
export declare const perlinNoisePresets: PerlinNoisePreset[];
export declare const PerlinNoise: React.FC<PerlinNoiseProps>;
export {};
