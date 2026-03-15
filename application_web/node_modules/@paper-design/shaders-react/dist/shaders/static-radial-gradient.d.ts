import { type ShaderComponentProps } from '../shader-mount.js';
import { type StaticRadialGradientParams, type ShaderPreset } from '@paper-design/shaders';
export interface StaticRadialGradientProps extends ShaderComponentProps, StaticRadialGradientParams {
}
type StaticRadialGradientPreset = ShaderPreset<StaticRadialGradientParams>;
export declare const defaultPreset: StaticRadialGradientPreset;
export declare const crossSectionPreset: StaticRadialGradientPreset;
export declare const radialPreset: StaticRadialGradientPreset;
export declare const loFiPreset: StaticRadialGradientPreset;
export declare const staticRadialGradientPresets: StaticRadialGradientPreset[];
export declare const StaticRadialGradient: React.FC<StaticRadialGradientProps>;
export {};
