import { type ShaderComponentProps } from '../shader-mount.js';
import { type MetaballsParams, type ShaderPreset } from '@paper-design/shaders';
export interface MetaballsProps extends ShaderComponentProps, MetaballsParams {
}
type MetaballsPreset = ShaderPreset<MetaballsParams>;
export declare const defaultPreset: MetaballsPreset;
export declare const inkDropsPreset: MetaballsPreset;
export declare const backgroundPreset: MetaballsPreset;
export declare const solarPreset: MetaballsPreset;
export declare const metaballsPresets: MetaballsPreset[];
export declare const Metaballs: React.FC<MetaballsProps>;
export {};
