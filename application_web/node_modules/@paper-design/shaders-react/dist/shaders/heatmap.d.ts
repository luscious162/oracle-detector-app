import React from 'react';
import { type ShaderComponentProps } from '../shader-mount.js';
import { type HeatmapParams, type ImageShaderPreset } from '@paper-design/shaders';
export interface HeatmapProps extends ShaderComponentProps, HeatmapParams {
    /**
     * Suspends the component when the image is being processed.
     */
    suspendWhenProcessingImage?: boolean;
}
export type HeatmapPreset = ImageShaderPreset<HeatmapParams>;
export declare const defaultPreset: HeatmapPreset;
export declare const sepiaPreset: HeatmapPreset;
export declare const heatmapPresets: HeatmapPreset[];
export declare const Heatmap: React.FC<HeatmapProps>;
