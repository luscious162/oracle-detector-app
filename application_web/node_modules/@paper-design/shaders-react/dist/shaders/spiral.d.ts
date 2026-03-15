import { type ShaderComponentProps } from '../shader-mount.js';
import { type ShaderPreset, type SpiralParams } from '@paper-design/shaders';
export interface SpiralProps extends ShaderComponentProps, SpiralParams {
}
type SpiralPreset = ShaderPreset<SpiralParams>;
export declare const defaultPreset: SpiralPreset;
export declare const dropletPreset: SpiralPreset;
export declare const junglePreset: SpiralPreset;
export declare const swirlPreset: SpiralPreset;
export declare const spiralPresets: SpiralPreset[];
export declare const Spiral: React.FC<SpiralProps>;
export {};
