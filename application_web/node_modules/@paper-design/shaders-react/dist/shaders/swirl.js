/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

import { memo } from "react";
import { ShaderMount } from "../shader-mount.js";
import {
  defaultObjectSizing,
  getShaderColorFromString,
  ShaderFitOptions,
  swirlFragmentShader
} from "@paper-design/shaders";
import { colorPropsAreEqual } from "../color-props-are-equal.js";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultObjectSizing,
    speed: 0.32,
    frame: 0,
    colorBack: "#330000",
    colors: ["#ffd1d1", "#ff8a8a", "#660000"],
    bandCount: 4,
    twist: 0.1,
    center: 0.2,
    proportion: 0.5,
    softness: 0,
    noiseFrequency: 0.4,
    noise: 0.2
  }
};
const openingPreset = {
  name: "Opening",
  params: {
    ...defaultObjectSizing,
    offsetX: -0.4,
    offsetY: 1,
    speed: 0.5,
    frame: 0,
    colorBack: "#ff8b61",
    colors: ["#fefff0", "#ffd8bd", "#ff8b61"],
    bandCount: 2,
    twist: 0.3,
    center: 0.2,
    proportion: 0.5,
    softness: 0,
    noiseFrequency: 0,
    noise: 0,
    scale: 1
  }
};
const jamesBondPreset = {
  name: "007",
  params: {
    ...defaultObjectSizing,
    speed: 1,
    frame: 0,
    colorBack: "#E9E7DA",
    colors: ["#000000"],
    bandCount: 5,
    twist: 0.3,
    center: 0,
    proportion: 0,
    softness: 0,
    noiseFrequency: 0.5,
    noise: 0
  }
};
const candyPreset = {
  name: "Candy",
  params: {
    ...defaultObjectSizing,
    speed: 1,
    frame: 0,
    colorBack: "#ffcd66",
    colors: ["#6bbceb", "#d7b3ff", "#ff9fff"],
    bandCount: 2,
    twist: 0.15,
    center: 0.2,
    proportion: 0.5,
    softness: 1,
    noiseFrequency: 0.5,
    noise: 0
  }
};
const swirlPresets = [defaultPreset, jamesBondPreset, openingPreset, candyPreset];
const Swirl = memo(function SwirlImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorBack = defaultPreset.params.colorBack,
  colors = defaultPreset.params.colors,
  bandCount = defaultPreset.params.bandCount,
  twist = defaultPreset.params.twist,
  center = defaultPreset.params.center,
  proportion = defaultPreset.params.proportion,
  softness = defaultPreset.params.softness,
  noiseFrequency = defaultPreset.params.noiseFrequency,
  noise = defaultPreset.params.noise,
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
    u_colors: colors.map(getShaderColorFromString),
    u_colorsCount: colors.length,
    u_bandCount: bandCount,
    u_twist: twist,
    u_center: center,
    u_proportion: proportion,
    u_softness: softness,
    u_noiseFrequency: noiseFrequency,
    u_noise: noise,
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
  return /* @__PURE__ */ jsx(ShaderMount, { ...props, speed, frame, fragmentShader: swirlFragmentShader, uniforms });
}, colorPropsAreEqual);
export {
  Swirl,
  candyPreset,
  defaultPreset,
  jamesBondPreset,
  openingPreset,
  swirlPresets
};
//# sourceMappingURL=swirl.js.map
