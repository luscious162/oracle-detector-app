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
  staticMeshGradientFragmentShader,
  ShaderFitOptions
} from "@paper-design/shaders";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultObjectSizing,
    rotation: 270,
    speed: 0,
    frame: 0,
    colors: ["#ffad0a", "#6200ff", "#e2a3ff", "#ff99fd"],
    positions: 2,
    waveX: 1,
    waveXShift: 0.6,
    waveY: 1,
    waveYShift: 0.21,
    mixing: 0.93,
    grainMixer: 0,
    grainOverlay: 0
  }
};
const seaPreset = {
  name: "Sea",
  params: {
    ...defaultObjectSizing,
    speed: 0,
    frame: 0,
    colors: ["#013b65", "#03738c", "#a3d3ff", "#f2faef"],
    positions: 0,
    waveX: 0.53,
    waveXShift: 0,
    waveY: 0.95,
    waveYShift: 0.64,
    mixing: 0.5,
    grainMixer: 0,
    grainOverlay: 0
  }
};
const sixtiesPreset = {
  name: "1960s",
  params: {
    ...defaultObjectSizing,
    speed: 0,
    frame: 0,
    colors: ["#000000", "#082400", "#b1aa91", "#8e8c15"],
    positions: 42,
    waveX: 0.45,
    waveXShift: 0,
    waveY: 1,
    waveYShift: 0,
    mixing: 0,
    grainMixer: 0.37,
    grainOverlay: 0.78
  }
};
const sunsetPreset = {
  name: "Sunset",
  params: {
    ...defaultObjectSizing,
    speed: 0,
    frame: 0,
    colors: ["#264653", "#9c2b2b", "#f4a261", "#ffffff"],
    positions: 0,
    waveX: 0.6,
    waveXShift: 0.7,
    waveY: 0.7,
    waveYShift: 0.7,
    mixing: 0.5,
    grainMixer: 0,
    grainOverlay: 0
  }
};
const staticMeshGradientPresets = [
  defaultPreset,
  sixtiesPreset,
  sunsetPreset,
  seaPreset
];
const StaticMeshGradient = memo(function StaticMeshGradientImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  colors = defaultPreset.params.colors,
  positions = defaultPreset.params.positions,
  waveX = defaultPreset.params.waveX,
  waveXShift = defaultPreset.params.waveXShift,
  waveY = defaultPreset.params.waveY,
  waveYShift = defaultPreset.params.waveYShift,
  mixing = defaultPreset.params.mixing,
  grainMixer = defaultPreset.params.grainMixer,
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
    u_colors: colors.map(getShaderColorFromString),
    u_colorsCount: colors.length,
    u_positions: positions,
    u_waveX: waveX,
    u_waveXShift: waveXShift,
    u_waveY: waveY,
    u_waveYShift: waveYShift,
    u_mixing: mixing,
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
      fragmentShader: staticMeshGradientFragmentShader,
      uniforms
    }
  );
}, colorPropsAreEqual);
export {
  StaticMeshGradient,
  defaultPreset,
  seaPreset,
  sixtiesPreset,
  staticMeshGradientPresets,
  sunsetPreset
};
//# sourceMappingURL=static-mesh-gradient.js.map
