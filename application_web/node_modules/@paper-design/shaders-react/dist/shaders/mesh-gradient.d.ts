import { type ShaderComponentProps } from '../shader-mount.js';
import { type MeshGradientParams, type ShaderPreset } from '@paper-design/shaders';
export interface MeshGradientProps extends ShaderComponentProps, MeshGradientParams {
}
type MeshGradientPreset = ShaderPreset<MeshGradientParams>;
export declare const defaultPreset: MeshGradientPreset;
export declare const purplePreset: MeshGradientPreset;
export declare const beachPreset: MeshGradientPreset;
export declare const inkPreset: MeshGradientPreset;
export declare const meshGradientPresets: MeshGradientPreset[];
export declare const MeshGradient: React.FC<MeshGradientProps>;
export {};
