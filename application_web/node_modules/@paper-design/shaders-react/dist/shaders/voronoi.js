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
  getShaderNoiseTexture,
  voronoiFragmentShader,
  ShaderFitOptions
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultPatternSizing,
    speed: 0.5,
    frame: 0,
    colors: ["#ff8247", "#ffe53d"],
    stepsPerColor: 3,
    colorGlow: "#ffffff",
    colorGap: "#2e0000",
    distortion: 0.4,
    gap: 0.04,
    glow: 0,
    scale: 0.5
  }
};
const cellsPreset = {
  name: "Cells",
  params: {
    ...defaultPatternSizing,
    scale: 0.5,
    speed: 0.5,
    frame: 0,
    colors: ["#ffffff"],
    stepsPerColor: 1,
    colorGlow: "#ffffff",
    colorGap: "#000000",
    distortion: 0.5,
    gap: 0.03,
    glow: 0.8
  }
};
const bubblesPreset = {
  name: "Bubbles",
  params: {
    ...defaultPatternSizing,
    scale: 0.75,
    speed: 0.5,
    frame: 0,
    colors: ["#83c9fb"],
    stepsPerColor: 1,
    colorGlow: "#ffffff",
    colorGap: "#ffffff",
    distortion: 0.4,
    gap: 0,
    glow: 1
  }
};
const lightsPreset = {
  name: "Lights",
  params: {
    ...defaultPatternSizing,
    scale: 3.3,
    speed: 0.5,
    frame: 0,
    colors: ["#fffffffc", "#bbff00", "#00ffff"],
    colorGlow: "#ff00d0",
    colorGap: "#ff00d0",
    stepsPerColor: 2,
    distortion: 0.38,
    gap: 0,
    glow: 1
  }
};
const voronoiPresets = [defaultPreset, lightsPreset, cellsPreset, bubblesPreset];
const Voronoi = memo(function VoronoiImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colors = defaultPreset.params.colors,
  stepsPerColor = defaultPreset.params.stepsPerColor,
  colorGlow = defaultPreset.params.colorGlow,
  colorGap = defaultPreset.params.colorGap,
  distortion = defaultPreset.params.distortion,
  gap = defaultPreset.params.gap,
  glow = defaultPreset.params.glow,
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
    u_colorGlow: getShaderColorFromString(colorGlow),
    u_colorGap: getShaderColorFromString(colorGap),
    u_distortion: distortion,
    u_gap: gap,
    u_glow: glow,
    u_noiseTexture: getShaderNoiseTexture(),
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
  return /* @__PURE__ */ jsx(ShaderMount, { ...props, speed, frame, fragmentShader: voronoiFragmentShader, uniforms });
}, colorPropsAreEqual);
export {
  Voronoi,
  bubblesPreset,
  cellsPreset,
  defaultPreset,
  lightsPreset,
  voronoiPresets
};
//# sourceMappingURL=voronoi.js.map
