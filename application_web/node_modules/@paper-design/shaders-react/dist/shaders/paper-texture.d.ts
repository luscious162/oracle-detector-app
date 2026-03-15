import { type ShaderComponentProps } from '../shader-mount.js';
import { type ImageShaderPreset, type PaperTextureParams } from '@paper-design/shaders';
export interface PaperTextureProps extends ShaderComponentProps, PaperTextureParams {
    /** @deprecated use `fiberSize` instead */
    fiberScale?: number;
    /** @deprecated use `crumpleSize` instead */
    crumplesScale?: number;
    /** @deprecated use `foldCount` instead */
    foldsNumber?: number;
    /** @deprecated use `fade` instead */
    blur?: number;
}
type PaperTexturePreset = ImageShaderPreset<PaperTextureParams>;
export declare const defaultPreset: PaperTexturePreset;
export declare const abstractPreset: PaperTexturePreset;
export declare const cardboardPreset: PaperTexturePreset;
export declare const detailsPreset: PaperTexturePreset;
export declare const paperTexturePresets: PaperTexturePreset[];
export declare const PaperTexture: React.FC<PaperTextureProps>;
export {};
