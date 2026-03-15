import { type ShaderComponentProps } from '../shader-mount.js';
import { type ShaderPreset, type SmokeRingParams } from '@paper-design/shaders';
export interface SmokeRingProps extends ShaderComponentProps, SmokeRingParams {
}
type SmokeRingPreset = ShaderPreset<SmokeRingParams>;
export declare const defaultPreset: SmokeRingPreset;
export declare const solarPreset: SmokeRingPreset;
export declare const linePreset: SmokeRingPreset;
export declare const cloudPreset: SmokeRingPreset;
export declare const smokeRingPresets: SmokeRingPreset[];
export declare const SmokeRing: React.FC<SmokeRingProps>;
export {};
