/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

import { memo } from "react";
import { ShaderMount } from "../shader-mount.js";
import {
  getShaderColorFromString,
  ditheringFragmentShader,
  ShaderFitOptions,
  defaultPatternSizing,
  defaultObjectSizing,
  DitheringTypes
} from "@paper-design/shaders";
import { DitheringShapes } from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultPatternSizing,
    speed: 1,
    frame: 0,
    scale: 0.6,
    colorBack: "#000000",
    colorFront: "#00b2ff",
    shape: "sphere",
    type: "4x4",
    size: 2
  }
};
const sinePreset = {
  name: "Sine Wave",
  params: {
    ...defaultPatternSizing,
    speed: 1,
    frame: 0,
    colorBack: "#730d54",
    colorFront: "#00becc",
    shape: "wave",
    type: "4x4",
    size: 11,
    scale: 1.2
  }
};
const bugsPreset = {
  name: "Bugs",
  params: {
    ...defaultPatternSizing,
    speed: 1,
    frame: 0,
    colorBack: "#000000",
    colorFront: "#008000",
    shape: "dots",
    type: "random",
    size: 9
  }
};
const ripplePreset = {
  name: "Ripple",
  params: {
    ...defaultObjectSizing,
    speed: 1,
    frame: 0,
    colorBack: "#603520",
    colorFront: "#c67953",
    shape: "ripple",
    type: "2x2",
    size: 3
  }
};
const swirlPreset = {
  name: "Swirl",
  params: {
    ...defaultObjectSizing,
    speed: 1,
    frame: 0,
    colorBack: "#00000000",
    colorFront: "#47a8e1",
    shape: "swirl",
    type: "8x8",
    size: 2
  }
};
const warpPreset = {
  name: "Warp",
  params: {
    ...defaultObjectSizing,
    speed: 1,
    frame: 0,
    colorBack: "#301c2a",
    colorFront: "#56ae6c",
    shape: "warp",
    type: "4x4",
    size: 2.5
  }
};
const ditheringPresets = [
  defaultPreset,
  warpPreset,
  sinePreset,
  ripplePreset,
  bugsPreset,
  swirlPreset
];
const Dithering = memo(function DitheringImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorBack = defaultPreset.params.colorBack,
  colorFront = defaultPreset.params.colorFront,
  shape = defaultPreset.params.shape,
  type = defaultPreset.params.type,
  pxSize,
  size = pxSize === void 0 ? defaultPreset.params.size : pxSize,
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
    u_colorBack: getShaderColorFromString(colorBack),
    u_colorFront: getShaderColorFromString(colorFront),
    u_shape: DitheringShapes[shape],
    u_type: DitheringTypes[type],
    u_pxSize: size,
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
  return /* @__PURE__ */ jsx(ShaderMount, { ...props, speed, frame, fragmentShader: ditheringFragmentShader, uniforms });
});
export {
  Dithering,
  bugsPreset,
  defaultPreset,
  ditheringPresets,
  ripplePreset,
  sinePreset,
  swirlPreset,
  warpPreset
};
//# sourceMappingURL=dithering.js.map
