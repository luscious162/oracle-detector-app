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
  ShaderFitOptions,
  spiralFragmentShader
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultPatternSizing,
    scale: 1,
    colorBack: "#001429",
    colorFront: "#79D1FF",
    density: 1,
    distortion: 0,
    strokeWidth: 0.5,
    strokeTaper: 0,
    strokeCap: 0,
    noise: 0,
    noiseFrequency: 0,
    softness: 0,
    speed: 1,
    frame: 0
  }
};
const dropletPreset = {
  name: "Droplet",
  params: {
    ...defaultPatternSizing,
    colorBack: "#effafe",
    colorFront: "#bf40a0",
    density: 0.9,
    distortion: 0,
    strokeWidth: 0.75,
    strokeTaper: 0.18,
    strokeCap: 1,
    noise: 0.74,
    noiseFrequency: 0.33,
    softness: 0.02,
    speed: 1,
    frame: 0
  }
};
const junglePreset = {
  name: "Jungle",
  params: {
    ...defaultPatternSizing,
    scale: 1.3,
    density: 0.5,
    colorBack: "#a0ef2a",
    colorFront: "#288b18",
    distortion: 0,
    strokeWidth: 0.5,
    strokeTaper: 0,
    strokeCap: 0,
    noise: 1,
    noiseFrequency: 0.25,
    softness: 0,
    speed: 0.75,
    frame: 0
  }
};
const swirlPreset = {
  name: "Swirl",
  params: {
    ...defaultPatternSizing,
    scale: 0.45,
    colorBack: "#b3e6d9",
    colorFront: "#1a2b4d",
    density: 0.2,
    distortion: 0,
    strokeWidth: 0.5,
    strokeTaper: 0,
    strokeCap: 0,
    noise: 0,
    noiseFrequency: 0.3,
    softness: 0.5,
    speed: 1,
    frame: 0
  }
};
const spiralPresets = [defaultPreset, junglePreset, dropletPreset, swirlPreset];
const Spiral = memo(function SpiralImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorBack = defaultPreset.params.colorBack,
  colorFront = defaultPreset.params.colorFront,
  density = defaultPreset.params.density,
  distortion = defaultPreset.params.distortion,
  strokeWidth = defaultPreset.params.strokeWidth,
  strokeTaper = defaultPreset.params.strokeTaper,
  strokeCap = defaultPreset.params.strokeCap,
  noiseFrequency = defaultPreset.params.noiseFrequency,
  noise = defaultPreset.params.noise,
  softness = defaultPreset.params.softness,
  // Sizing props
  fit = defaultPreset.params.fit,
  rotation = defaultPreset.params.rotation,
  scale = defaultPreset.params.scale,
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
    u_colorBack: getShaderColorFromString(colorBack),
    u_colorFront: getShaderColorFromString(colorFront),
    u_density: density,
    u_distortion: distortion,
    u_strokeWidth: strokeWidth,
    u_strokeTaper: strokeTaper,
    u_strokeCap: strokeCap,
    u_noiseFrequency: noiseFrequency,
    u_noise: noise,
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
  return /* @__PURE__ */ jsx(ShaderMount, { ...props, speed, frame, fragmentShader: spiralFragmentShader, uniforms });
}, colorPropsAreEqual);
export {
  Spiral,
  defaultPreset,
  dropletPreset,
  junglePreset,
  spiralPresets,
  swirlPreset
};
//# sourceMappingURL=spiral.js.map
