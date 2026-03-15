import { type ShaderComponentProps } from '../shader-mount.js';
import { type ShaderPreset, type SwirlParams } from '@paper-design/shaders';
export interface SwirlProps extends ShaderComponentProps, SwirlParams {
}
type SwirlPreset = ShaderPreset<SwirlParams>;
export declare const defaultPreset: SwirlPreset;
export declare const openingPreset: SwirlPreset;
export declare const jamesBondPreset: SwirlPreset;
export declare const candyPreset: SwirlPreset;
export declare const swirlPresets: SwirlPreset[];
export declare const Swirl: React.FC<SwirlProps>;
export {};
