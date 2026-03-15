import { type ShaderComponentProps } from '../shader-mount.js';
import { type DitheringParams, type ShaderPreset } from '@paper-design/shaders';
export interface DitheringProps extends ShaderComponentProps, DitheringParams {
    /** @deprecated use `size` instead */
    pxSize?: number;
}
type DitheringPreset = ShaderPreset<DitheringParams>;
export declare const defaultPreset: DitheringPreset;
export declare const sinePreset: DitheringPreset;
export declare const bugsPreset: DitheringPreset;
export declare const ripplePreset: DitheringPreset;
export declare const swirlPreset: DitheringPreset;
export declare const warpPreset: DitheringPreset;
export declare const ditheringPresets: DitheringPreset[];
export declare const Dithering: React.FC<DitheringProps>;
export {};
