import { type ShaderComponentProps } from '../shader-mount.js';
import { type SimplexNoiseParams, type ShaderPreset } from '@paper-design/shaders';
export interface SimplexNoiseProps extends ShaderComponentProps, SimplexNoiseParams {
}
type SimplexNoisePreset = ShaderPreset<SimplexNoiseParams>;
export declare const defaultPreset: SimplexNoisePreset;
export declare const bubblegumPreset: SimplexNoisePreset;
export declare const spotsPreset: SimplexNoisePreset;
export declare const firstContactPreset: SimplexNoisePreset;
export declare const simplexNoisePresets: SimplexNoisePreset[];
export declare const SimplexNoise: React.FC<SimplexNoiseProps>;
export {};
