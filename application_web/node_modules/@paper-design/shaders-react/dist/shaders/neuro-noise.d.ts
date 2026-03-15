import { type ShaderComponentProps } from '../shader-mount.js';
import { type NeuroNoiseParams, type ShaderPreset } from '@paper-design/shaders';
export interface NeuroNoiseProps extends ShaderComponentProps, NeuroNoiseParams {
}
type NeuroNoisePreset = ShaderPreset<NeuroNoiseParams>;
export declare const defaultPreset: NeuroNoisePreset;
export declare const sensationPreset: NeuroNoisePreset;
export declare const bloodstreamPreset: NeuroNoisePreset;
export declare const ghostPreset: NeuroNoisePreset;
export declare const neuroNoisePresets: NeuroNoisePreset[];
export declare const NeuroNoise: React.FC<NeuroNoiseProps>;
export {};
