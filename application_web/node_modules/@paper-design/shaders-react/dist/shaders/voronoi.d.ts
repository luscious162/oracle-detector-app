import { type ShaderComponentProps } from '../shader-mount.js';
import { type VoronoiParams, type ShaderPreset } from '@paper-design/shaders';
export interface VoronoiProps extends ShaderComponentProps, VoronoiParams {
}
type VoronoiPreset = ShaderPreset<VoronoiParams>;
export declare const defaultPreset: VoronoiPreset;
export declare const cellsPreset: VoronoiPreset;
export declare const bubblesPreset: VoronoiPreset;
export declare const lightsPreset: VoronoiPreset;
export declare const voronoiPresets: VoronoiPreset[];
export declare const Voronoi: React.FC<VoronoiProps>;
export {};
