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
  staticRadialGradientFragmentShader,
  ShaderFitOptions
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultObjectSizing,
    scale: 1,
    speed: 0,
    frame: 0,
    colorBack: "#000000",
    colors: ["#00bbff", "#00ffe1", "#ffffff"],
    radius: 0.8,
    focalDistance: 0.99,
    focalAngle: 0,
    falloff: 0.24,
    mixing: 0.5,
    distortion: 0,
    distortionShift: 0,
    distortionFreq: 12,
    grainMixer: 0,
    grainOverlay: 0
  }
};
const crossSectionPreset = {
  name: "Cross Section",
  params: {
    ...defaultObjectSizing,
    scale: 1,
    speed: 0,
    frame: 0,
    colorBack: "#3d348b",
    colors: ["#7678ed", "#f7b801", "#f18701", "#37a066"],
    radius: 1,
    focalDistance: 0,
    focalAngle: 0,
    falloff: 0,
    mixing: 0,
    distortion: 1,
    distortionShift: 0,
    distortionFreq: 12,
    grainMixer: 0,
    grainOverlay: 0
  }
};
const radialPreset = {
  name: "Radial",
  params: {
    ...defaultObjectSizing,
    scale: 1,
    speed: 0,
    frame: 0,
    colorBack: "#264653",
    colors: ["#9c2b2b", "#f4a261", "#ffffff"],
    radius: 1,
    focalDistance: 0,
    focalAngle: 0,
    falloff: 0,
    mixing: 1,
    distortion: 0,
    distortionShift: 0,
    distortionFreq: 12,
    grainMixer: 0,
    grainOverlay: 0
  }
};
const loFiPreset = {
  name: "Lo-Fi",
  params: {
    ...defaultObjectSizing,
    speed: 0,
    frame: 0,
    colorBack: "#2e1f27",
    colors: ["#d72638", "#3f88c5", "#f49d37"],
    radius: 1,
    focalDistance: 0,
    focalAngle: 0,
    falloff: 0.9,
    mixing: 0.7,
    distortion: 0,
    distortionShift: 0,
    distortionFreq: 12,
    grainMixer: 1,
    grainOverlay: 0.5
  }
};
const staticRadialGradientPresets = [
  defaultPreset,
  loFiPreset,
  crossSectionPreset,
  radialPreset
];
const StaticRadialGradient = memo(function StaticRadialGradientImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colorBack = defaultPreset.params.colorBack,
  colors = defaultPreset.params.colors,
  radius = defaultPreset.params.radius,
  focalDistance = defaultPreset.params.focalDistance,
  focalAngle = defaultPreset.params.focalAngle,
  falloff = defaultPreset.params.falloff,
  grainMixer = defaultPreset.params.grainMixer,
  mixing = defaultPreset.params.mixing,
  distortion = defaultPreset.params.distortion,
  distortionShift = defaultPreset.params.distortionShift,
  distortionFreq = defaultPreset.params.distortionFreq,
  grainOverlay = defaultPreset.params.grainOverlay,
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
    u_radius: radius,
    u_focalDistance: focalDistance,
    u_focalAngle: focalAngle,
    u_falloff: falloff,
    u_mixing: mixing,
    u_distortion: distortion,
    u_distortionShift: distortionShift,
    u_distortionFreq: distortionFreq,
    u_grainMixer: grainMixer,
    u_grainOverlay: grainOverlay,
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
      fragmentShader: staticRadialGradientFragmentShader,
      uniforms
    }
  );
}, colorPropsAreEqual);
export {
  StaticRadialGradient,
  crossSectionPreset,
  defaultPreset,
  loFiPreset,
  radialPreset,
  staticRadialGradientPresets
};
//# sourceMappingURL=static-radial-gradient.js.map
