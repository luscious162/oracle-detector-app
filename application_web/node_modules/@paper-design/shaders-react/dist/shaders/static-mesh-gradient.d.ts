import { type ShaderComponentProps } from '../shader-mount.js';
import { type StaticMeshGradientParams, type ShaderPreset } from '@paper-design/shaders';
export interface StaticMeshGradientProps extends ShaderComponentProps, StaticMeshGradientParams {
}
type StaticMeshGradientPreset = ShaderPreset<StaticMeshGradientParams>;
export declare const defaultPreset: StaticMeshGradientPreset;
export declare const seaPreset: StaticMeshGradientPreset;
export declare const sixtiesPreset: StaticMeshGradientPreset;
export declare const sunsetPreset: StaticMeshGradientPreset;
export declare const staticMeshGradientPresets: StaticMeshGradientPreset[];
export declare const StaticMeshGradient: React.FC<StaticMeshGradientProps>;
export {};
