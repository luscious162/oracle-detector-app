/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

import { memo } from "react";
import { ShaderMount } from "../shader-mount.js";
import { colorPropsAreEqual } from "../color-props-are-equal.js";
import {
  defaultPatternSizing,
  getShaderColorFromString,
  neuroNoiseFragmentShader,
  ShaderFitOptions
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultPatternSizing,
    speed: 1,
    frame: 0,
    colorFront: "#ffffff",
    colorMid: "#47a6ff",
    colorBack: "#000000",
    brightness: 0.05,
    contrast: 0.3
  }
};
const sensationPreset = {
  name: "Sensation",
  params: {
    ...defaultPatternSizing,
    speed: 1,
    frame: 0,
    colorFront: "#00c8ff",
    colorMid: "#fbff00",
    colorBack: "#8b42ff",
    brightness: 0.19,
    contrast: 0.12,
    scale: 3
  }
};
const bloodstreamPreset = {
  name: "Bloodstream",
  params: {
    ...defaultPatternSizing,
    speed: 1,
    frame: 0,
    colorFront: "#ff0000",
    colorMid: "#ff0000",
    colorBack: "#ffffff",
    brightness: 0.24,
    contrast: 0.17,
    scale: 0.7
  }
};
const ghostPreset = {
  name: "Ghost",
  params: {
    ...defaultPatternSizing,
    speed: 1,
    frame: 0,
    colorFront: "#ffffff",
    colorMid: "#000000",
    colorBack: "#ffffff",
    brightness: 0,
    contrast: 1,
    scale: 0.55
  }
};
const neuroNoisePresets = [
  defaultPreset,
  sensationPreset,
  bloodstreamPreset,
  ghostPreset
];
const NeuroNoise = memo(function NeuroNoiseImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorFront = defaultPreset.params.colorFront,
  colorMid = defaultPreset.params.colorMid,
  colorBack = defaultPreset.params.colorBack,
  brightness = defaultPreset.params.brightness,
  contrast = defaultPreset.params.contrast,
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
    u_colorFront: getShaderColorFromString(colorFront),
    u_colorMid: getShaderColorFromString(colorMid),
    u_colorBack: getShaderColorFromString(colorBack),
    u_brightness: brightness,
    u_contrast: contrast,
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
  return /* @__PURE__ */ jsx(ShaderMount, { ...props, speed, frame, fragmentShader: neuroNoiseFragmentShader, uniforms });
}, colorPropsAreEqual);
export {
  NeuroNoise,
  bloodstreamPreset,
  defaultPreset,
  ghostPreset,
  neuroNoisePresets,
  sensationPreset
};
//# sourceMappingURL=neuro-noise.js.map
