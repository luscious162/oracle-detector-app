import { type ShaderComponentProps } from '../shader-mount.js';
import { type WaterParams, type ImageShaderPreset } from '@paper-design/shaders';
export interface WaterProps extends ShaderComponentProps, WaterParams {
    /** @deprecated use `size` instead */
    effectScale?: number;
}
type WaterPreset = ImageShaderPreset<WaterParams>;
export declare const defaultPreset: WaterPreset;
export declare const abstractPreset: WaterPreset;
export declare const streamingPreset: WaterPreset;
export declare const slowMoPreset: WaterPreset;
export declare const waterPresets: WaterPreset[];
export declare const Water: React.FC<WaterProps>;
export {};
