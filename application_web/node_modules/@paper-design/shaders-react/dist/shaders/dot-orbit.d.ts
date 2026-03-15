import { type ShaderComponentProps } from '../shader-mount.js';
import { type DotOrbitParams, type ShaderPreset } from '@paper-design/shaders';
export interface DotOrbitProps extends ShaderComponentProps, DotOrbitParams {
}
type DotOrbitPreset = ShaderPreset<DotOrbitParams>;
export declare const defaultPreset: DotOrbitPreset;
export declare const shinePreset: DotOrbitPreset;
export declare const bubblesPreset: DotOrbitPreset;
export declare const hallucinatoryPreset: DotOrbitPreset;
export declare const dotOrbitPresets: DotOrbitPreset[];
export declare const DotOrbit: React.FC<DotOrbitProps>;
export {};
