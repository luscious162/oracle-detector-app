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
  paperTextureFragmentShader,
  ShaderFitOptions
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultObjectSizing,
    fit: "cover",
    scale: 0.6,
    speed: 0,
    frame: 0,
    colorFront: "#9fadbc",
    colorBack: "#ffffff",
    contrast: 0.3,
    roughness: 0.4,
    fiber: 0.3,
    fiberSize: 0.2,
    crumples: 0.3,
    crumpleSize: 0.35,
    folds: 0.65,
    foldCount: 5,
    fade: 0,
    drops: 0.2,
    seed: 5.8
  }
};
const abstractPreset = {
  name: "Abstract",
  params: {
    ...defaultObjectSizing,
    fit: "cover",
    speed: 0,
    frame: 0,
    scale: 0.6,
    colorFront: "#00eeff",
    colorBack: "#ff0a81",
    contrast: 0.85,
    roughness: 0,
    fiber: 0.1,
    fiberSize: 0.2,
    crumples: 0,
    crumpleSize: 0.3,
    folds: 1,
    foldCount: 3,
    fade: 0,
    drops: 0.2,
    seed: 2.2
  }
};
const cardboardPreset = {
  name: "Cardboard",
  params: {
    ...defaultObjectSizing,
    fit: "cover",
    speed: 0,
    frame: 0,
    scale: 0.6,
    colorFront: "#c7b89e",
    colorBack: "#999180",
    contrast: 0.4,
    roughness: 0,
    fiber: 0.35,
    fiberSize: 0.14,
    crumples: 0.7,
    crumpleSize: 0.1,
    folds: 0,
    foldCount: 1,
    fade: 0,
    drops: 0.1,
    seed: 1.6
  }
};
const detailsPreset = {
  name: "Details",
  params: {
    ...defaultObjectSizing,
    speed: 0,
    frame: 0,
    fit: "cover",
    scale: 3,
    colorFront: "#00000000",
    colorBack: "#00000000",
    contrast: 0,
    roughness: 1,
    fiber: 0.27,
    fiberSize: 0.22,
    crumples: 1,
    crumpleSize: 0.5,
    folds: 1,
    foldCount: 15,
    fade: 0,
    drops: 0,
    seed: 6
  }
};
const paperTexturePresets = [
  defaultPreset,
  cardboardPreset,
  abstractPreset,
  detailsPreset
];
const PaperTexture = memo(function PaperTextureImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorFront = defaultPreset.params.colorFront,
  colorBack = defaultPreset.params.colorBack,
  image = "",
  contrast = defaultPreset.params.contrast,
  roughness = defaultPreset.params.roughness,
  fiber = defaultPreset.params.fiber,
  crumples = defaultPreset.params.crumples,
  folds = defaultPreset.params.folds,
  drops = defaultPreset.params.drops,
  seed = defaultPreset.params.seed,
  // Reworked props
  fiberScale,
  fiberSize = fiberScale === void 0 ? defaultPreset.params.fiberSize : 0.2 / fiberScale,
  crumplesScale,
  crumpleSize = crumplesScale === void 0 ? defaultPreset.params.crumpleSize : 0.2 / crumplesScale,
  blur,
  fade = blur === void 0 ? defaultPreset.params.fade : blur,
  foldsNumber,
  foldCount = foldsNumber === void 0 ? defaultPreset.params.foldCount : foldsNumber,
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
  const noiseTexture = typeof window !== "undefined" && { u_noiseTexture: getShaderNoiseTexture() };
  const uniforms = {
    // Own uniforms
    u_image: image,
    u_colorFront: getShaderColorFromString(colorFront),
    u_colorBack: getShaderColorFromString(colorBack),
    u_contrast: contrast,
    u_roughness: roughness,
    u_fiber: fiber,
    u_fiberSize: fiberSize,
    u_crumples: crumples,
    u_crumpleSize: crumpleSize,
    u_foldCount: foldCount,
    u_folds: folds,
    u_fade: fade,
    u_drops: drops,
    u_seed: seed,
    ...noiseTexture,
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
      fragmentShader: paperTextureFragmentShader,
      mipmaps: ["u_image"],
      uniforms
    }
  );
}, colorPropsAreEqual);
export {
  PaperTexture,
  abstractPreset,
  cardboardPreset,
  defaultPreset,
  detailsPreset,
  paperTexturePresets
};
//# sourceMappingURL=paper-texture.js.map
