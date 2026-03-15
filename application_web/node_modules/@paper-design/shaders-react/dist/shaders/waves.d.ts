import { type ShaderComponentProps } from '../shader-mount.js';
import { type WavesParams, type ShaderPreset } from '@paper-design/shaders';
export interface WavesProps extends ShaderComponentProps, WavesParams {
}
type WavesPreset = ShaderPreset<WavesParams>;
export declare const defaultPreset: WavesPreset;
export declare const groovyPreset: WavesPreset;
export declare const tangledUpPreset: WavesPreset;
export declare const waveRidePreset: WavesPreset;
export declare const wavesPresets: WavesPreset[];
export declare const Waves: React.FC<WavesProps>;
export {};
