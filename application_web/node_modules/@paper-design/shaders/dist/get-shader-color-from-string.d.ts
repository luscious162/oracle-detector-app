/**  Convert color string from HSL, RGB, or hex to 0-to-1-range-RGBA array */
export declare function getShaderColorFromString(colorString: string | [number, number, number] | [number, number, number, number] | undefined): [number, number, number, number];
export declare const clamp: (n: number, min: number, max: number) => number;
