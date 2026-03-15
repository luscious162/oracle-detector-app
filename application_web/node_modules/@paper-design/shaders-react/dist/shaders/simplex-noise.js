/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

import { memo } from "react";
import { ShaderMount } from "../shader-mount.js";
import { colorPropsAreEqual } from "../color-props-are-equal.js";
import {
  getShaderColorFromString,
  simplexNoiseFragmentShader,
  ShaderFitOptions,
  defaultPatternSizing
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultPatternSizing,
    scale: 0.6,
    speed: 0.5,
    frame: 0,
    colors: ["#4449CF", "#FFD1E0", "#F94446", "#FFD36B", "#FFFFFF"],
    stepsPerColor: 2,
    softness: 0
  }
};
const bubblegumPreset = {
  name: "Bubblegum",
  params: {
    ...defaultPatternSizing,
    speed: 2,
    frame: 0,
    colors: ["#ffffff", "#ff9e9e", "#5f57ff", "#00f7ff"],
    stepsPerColor: 1,
    softness: 1,
    scale: 1.6
  }
};
const spotsPreset = {
  name: "Spots",
  params: {
    ...defaultPatternSizing,
    speed: 0.6,
    frame: 0,
    colors: ["#ff7b00", "#f9ffeb", "#320d82"],
    stepsPerColor: 1,
    softness: 0,
    scale: 1
  }
};
const firstContactPreset = {
  name: "First contact",
  params: {
    ...defaultPatternSizing,
    speed: 2,
    frame: 0,
    colors: ["#e8cce6", "#120d22", "#442c44", "#e6baba", "#fff5f5"],
    stepsPerColor: 2,
    softness: 0,
    scale: 0.2
  }
};
const simplexNoisePresets = [
  defaultPreset,
  spotsPreset,
  firstContactPreset,
  bubblegumPreset
];
const SimplexNoise = memo(function SimplexNoiseImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colors = defaultPreset.params.colors,
  stepsPerColor = defaultPreset.params.stepsPerColor,
  softness = defaultPreset.params.softness,
  // Sizing props
  fit = defaultPreset.params.fit,
  scale = defaultPreset.params.scale,
  rotation = defaultPreset.params.rotation,
  originX = defaultPreset.params.originX,
  originY = defaultPreset.params.originY,
  offsetX = defaultPreset.params.offsetX,
  offsetY = defaultPreset.params.offsetY,
  worldWidth = defaultPreset.params.worldWidth,
  worldHeight = defaultPreset.params.worldHeight,
  ...props
}) {
  const uniforms = {
    // Own uniforms
    u_colors: colors.map(getShaderColorFromString),
    u_colorsCount: colors.length,
    u_stepsPerColor: stepsPerColor,
    u_softness: softness,
    // Sizing uniforms
    u_fit: ShaderFitOptions[fit],
    u_scale: scale,
    u_rotation: rotation,
    u_offsetX: offsetX,
    u_offsetY: offsetY,
    u_originX: originX,
    u_originY: originY,
    u_worldWidth: worldWidth,
    u_worldHeight: worldHeight
  };
  return /* @__PURE__ */ jsx(
    ShaderMount,
    {
      ...props,
      speed,
      frame,
      fragmentShader: simplexNoiseFragmentShader,
      uniforms
    }
  );
}, colorPropsAreEqual);
export {
  SimplexNoise,
  bubblegumPreset,
  defaultPreset,
  firstContactPreset,
  simplexNoisePresets,
  spotsPreset
};
//# sourceMappingURL=simplex-noise.js.map
