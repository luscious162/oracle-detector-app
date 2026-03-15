import { type ShaderComponentProps } from '../shader-mount.js';
import { type WarpParams, type ShaderPreset } from '@paper-design/shaders';
export interface WarpProps extends ShaderComponentProps, WarpParams {
}
type WarpPreset = ShaderPreset<WarpParams>;
export declare const defaultPreset: WarpPreset;
export declare const presetCauldron: WarpPreset;
export declare const presetInk: WarpPreset;
export declare const presetKelp: WarpPreset;
export declare const presetNectar: WarpPreset;
export declare const presetPassion: WarpPreset;
export declare const warpPresets: WarpPreset[];
export declare const Warp: React.FC<WarpProps>;
export {};
