import { type ShaderComponentProps } from '../shader-mount.js';
import { type DotGridParams, type ShaderPreset } from '@paper-design/shaders';
export interface DotGridProps extends ShaderComponentProps, DotGridParams {
}
type DotGridPreset = ShaderPreset<DotGridParams>;
export declare const defaultPreset: DotGridPreset;
export declare const dotGridPresets: DotGridPreset[];
export declare const DotGrid: React.FC<DotGridProps>;
export {};
