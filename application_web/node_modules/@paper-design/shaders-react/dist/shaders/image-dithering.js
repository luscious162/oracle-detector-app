/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

import { memo } from "react";
import { ShaderMount } from "../shader-mount.js";
import { colorPropsAreEqual } from "../color-props-are-equal.js";
import {
  imageDitheringFragmentShader,
  getShaderColorFromString,
  ShaderFitOptions,
  defaultObjectSizing,
  DitheringTypes
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultObjectSizing,
    fit: "cover",
    // scale: 0.95,
    speed: 0,
    frame: 0,
    colorFront: "#94ffaf",
    colorBack: "#000c38",
    colorHighlight: "#eaff94",
    type: "8x8",
    size: 2,
    colorSteps: 2,
    originalColors: false,
    inverted: false
  }
};
const retroPreset = {
  name: "Retro",
  params: {
    ...defaultObjectSizing,
    fit: "cover",
    speed: 0,
    frame: 0,
    colorFront: "#eeeeee",
    colorBack: "#5452ff",
    colorHighlight: "#eeeeee",
    type: "2x2",
    size: 3,
    colorSteps: 1,
    originalColors: true,
    inverted: false
  }
};
const noisePreset = {
  name: "Noise",
  params: {
    ...defaultObjectSizing,
    fit: "cover",
    speed: 0,
    frame: 0,
    colorFront: "#a2997c",
    colorBack: "#000000",
    colorHighlight: "#ededed",
    type: "random",
    size: 1,
    colorSteps: 1,
    originalColors: false,
    inverted: false
  }
};
const naturalPreset = {
  name: "Natural",
  params: {
    ...defaultObjectSizing,
    fit: "cover",
    speed: 0,
    frame: 0,
    colorFront: "#ffffff",
    colorBack: "#000000",
    colorHighlight: "#ffffff",
    type: "8x8",
    size: 2,
    colorSteps: 5,
    originalColors: true,
    inverted: false
  }
};
const imageDitheringPresets = [defaultPreset, noisePreset, retroPreset, naturalPreset];
const ImageDithering = memo(function ImageDitheringImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorFront = defaultPreset.params.colorFront,
  colorBack = defaultPreset.params.colorBack,
  colorHighlight = defaultPreset.params.colorHighlight,
  image = "",
  type = defaultPreset.params.type,
  colorSteps = defaultPreset.params.colorSteps,
  originalColors = defaultPreset.params.originalColors,
  inverted = defaultPreset.params.inverted,
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
    u_image: image,
    u_colorFront: getShaderColorFromString(colorFront),
    u_colorBack: getShaderColorFromString(colorBack),
    u_colorHighlight: getShaderColorFromString(colorHighlight),
    u_type: DitheringTypes[type],
    u_pxSize: size,
    u_colorSteps: colorSteps,
    u_originalColors: originalColors,
    u_inverted: inverted,
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
  return /* @__PURE__ */ jsx(
    ShaderMount,
    {
      ...props,
      speed,
      frame,
      fragmentShader: imageDitheringFragmentShader,
      uniforms
    }
  );
}, colorPropsAreEqual);
export {
  ImageDithering,
  defaultPreset,
  imageDitheringPresets,
  naturalPreset,
  noisePreset,
  retroPreset
};
//# sourceMappingURL=image-dithering.js.map
