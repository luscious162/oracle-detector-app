import { type ShaderComponentProps } from '../shader-mount.js';
import { type LiquidMetalParams, type ImageShaderPreset } from '@paper-design/shaders';
export interface LiquidMetalProps extends ShaderComponentProps, LiquidMetalParams {
    /**
     * Suspends the component when the image is being processed.
     */
    suspendWhenProcessingImage?: boolean;
}
type LiquidMetalPreset = ImageShaderPreset<LiquidMetalParams>;
export declare const defaultPreset: LiquidMetalPreset;
export declare const noirPreset: LiquidMetalPreset;
export declare const fullScreenPreset: LiquidMetalPreset;
export declare const stripesPreset: LiquidMetalPreset;
export declare const liquidMetalPresets: LiquidMetalPreset[];
export declare const LiquidMetal: React.FC<LiquidMetalProps>;
export {};
