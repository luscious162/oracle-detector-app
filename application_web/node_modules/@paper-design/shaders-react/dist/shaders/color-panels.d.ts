import { type ShaderComponentProps } from '../shader-mount.js';
import { type ColorPanelsParams, type ShaderPreset } from '@paper-design/shaders';
export interface ColorPanelsProps extends ShaderComponentProps, ColorPanelsParams {
}
type ColorPanelsPreset = ShaderPreset<ColorPanelsParams>;
export declare const defaultPreset: ColorPanelsPreset;
export declare const glassPreset: ColorPanelsPreset;
export declare const gradientPreset: ColorPanelsPreset;
export declare const openingPreset: ColorPanelsPreset;
export declare const colorPanelsPresets: ColorPanelsPreset[];
export declare const ColorPanels: React.FC<ColorPanelsProps>;
export {};
