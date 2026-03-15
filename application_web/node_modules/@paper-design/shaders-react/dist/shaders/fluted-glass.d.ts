import { type ShaderComponentProps } from '../shader-mount.js';
import { type FlutedGlassParams, type ImageShaderPreset } from '@paper-design/shaders';
export interface FlutedGlassProps extends ShaderComponentProps, FlutedGlassParams {
    /** @deprecated use `size` instead */
    count?: number;
}
type FlutedGlassPreset = ImageShaderPreset<FlutedGlassParams>;
export declare const defaultPreset: FlutedGlassPreset;
export declare const wavesPreset: FlutedGlassPreset;
export declare const abstractPreset: FlutedGlassPreset;
export declare const foldsPreset: FlutedGlassPreset;
export declare const flutedGlassPresets: FlutedGlassPreset[];
export declare const FlutedGlass: React.FC<FlutedGlassProps>;
export {};
