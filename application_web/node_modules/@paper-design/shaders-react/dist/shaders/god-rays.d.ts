import { type ShaderComponentProps } from '../shader-mount.js';
import { type GodRaysParams, type ShaderPreset } from '@paper-design/shaders';
export interface GodRaysProps extends ShaderComponentProps, GodRaysParams {
}
type GodRaysPreset = ShaderPreset<GodRaysParams>;
export declare const defaultPreset: GodRaysPreset;
export declare const warpPreset: GodRaysPreset;
export declare const linearPreset: GodRaysPreset;
export declare const etherPreset: GodRaysPreset;
export declare const godRaysPresets: GodRaysPreset[];
export declare const GodRays: React.FC<GodRaysProps>;
export {};
