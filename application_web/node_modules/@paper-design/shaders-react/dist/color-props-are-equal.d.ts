interface PropsWithColors {
    colors?: string[];
    [key: string]: unknown;
}
export declare function colorPropsAreEqual(prevProps: PropsWithColors, nextProps: PropsWithColors): boolean;
export {};
