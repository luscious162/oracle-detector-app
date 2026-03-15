/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

import { memo } from "react";
import { ShaderMount } from "../shader-mount.js";
import { colorPropsAreEqual } from "../color-props-are-equal.js";
import {
  defaultObjectSizing,
  getShaderColorFromString,
  getShaderNoiseTexture,
  metaballsFragmentShader,
  ShaderFitOptions
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultObjectSizing,
    scale: 1,
    speed: 1,
    frame: 0,
    colorBack: "#000000",
    colors: ["#6e33cc", "#ff5500", "#ffc105", "#ffc800", "#f585ff"],
    count: 10,
    size: 0.83
  }
};
const inkDropsPreset = {
  name: "Ink Drops",
  params: {
    ...defaultObjectSizing,
    scale: 1,
    speed: 2,
    frame: 0,
    colorBack: "#ffffff00",
    colors: ["#000000"],
    count: 18,
    size: 0.1
  }
};
const backgroundPreset = {
  name: "Background",
  params: {
    ...defaultObjectSizing,
    speed: 0.5,
    frame: 0,
    colors: ["#ae00ff", "#00ff95", "#ffc105"],
    colorBack: "#2a273f",
    count: 13,
    size: 0.81,
    scale: 4,
    rotation: 0,
    offsetX: -0.3
  }
};
const solarPreset = {
  name: "Solar",
  params: {
    ...defaultObjectSizing,
    speed: 1,
    frame: 0,
    colors: ["#ffc800", "#ff5500", "#ffc105"],
    colorBack: "#102f84",
    count: 7,
    size: 0.75,
    scale: 1
  }
};
const metaballsPresets = [defaultPreset, inkDropsPreset, solarPreset, backgroundPreset];
const Metaballs = memo(function MetaballsImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorBack = defaultPreset.params.colorBack,
  colors = defaultPreset.params.colors,
  size = defaultPreset.params.size,
  count = defaultPreset.params.count,
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
    u_size: size,
    u_count: count,
    u_noiseTexture: getShaderNoiseTexture(),
    // Sizing uniforms
    u_fit: ShaderFitOptions[fit],
    u_rotation: rotation,
    u_scale: scale,
    u_offsetX: offsetX,
    u_offsetY: offsetY,
    u_originX: originX,
    u_originY: originY,
    u_worldWidth: worldWidth,
    u_worldHeight: worldHeight
  };
  return /* @__PURE__ */ jsx(ShaderMount, { ...props, speed, frame, fragmentShader: metaballsFragmentShader, uniforms });
}, colorPropsAreEqual);
export {
  Metaballs,
  backgroundPreset,
  defaultPreset,
  inkDropsPreset,
  metaballsPresets,
  solarPreset
};
//# sourceMappingURL=metaballs.js.map
