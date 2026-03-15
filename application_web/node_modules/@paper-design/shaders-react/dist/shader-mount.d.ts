import { type PaperShaderElement, type ShaderMotionParams } from '@paper-design/shaders';
/**
 * React Shader Mount can also accept strings as uniform values, which will assumed to be URLs and loaded as images
 *
 * We accept undefined as a convenience for server rendering, when some things may be undefined
 * We just skip setting the uniform if it's undefined. This allows the shader mount to still take up space during server rendering
 */
interface ShaderMountUniformsReact {
    [key: string]: string | boolean | number | number[] | number[][] | HTMLImageElement | undefined;
}
export interface ShaderMountProps extends Omit<React.ComponentProps<'div'>, 'color' | 'ref'>, ShaderMotionParams {
    ref?: React.Ref<PaperShaderElement>;
    fragmentShader: string;
    uniforms: ShaderMountUniformsReact;
    mipmaps?: string[];
    minPixelRatio?: number;
    maxPixelCount?: number;
    webGlContextAttributes?: WebGLContextAttributes;
    /** Inline CSS width style */
    width?: string | number;
    /** Inline CSS height style */
    height?: string | number;
}
export interface ShaderComponentProps extends Omit<React.ComponentProps<'div'>, 'color' | 'ref'> {
    ref?: React.Ref<PaperShaderElement>;
    minPixelRatio?: number;
    maxPixelCount?: number;
    webGlContextAttributes?: WebGLContextAttributes;
    /** Inline CSS width style */
    width?: string | number;
    /** Inline CSS height style */
    height?: string | number;
}
/**
 * A React component that mounts a shader and updates its uniforms as the component's props change
 * If you pass a string as a uniform value, it will be assumed to be a URL and attempted to be loaded as an image
 */
export declare const ShaderMount: React.FC<ShaderMountProps>;
export {};
