/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

import React, { memo, useLayoutEffect, useMemo, useState } from "react";
import { ShaderMount } from "../shader-mount.js";
import {
  getShaderColorFromString,
  heatmapFragmentShader,
  ShaderFitOptions,
  defaultObjectSizing,
  toProcessedHeatmap
} from "@paper-design/shaders";
import { transparentPixel } from "../transparent-pixel.js";
import { suspend } from "../suspend.js";
import { colorPropsAreEqual } from "../color-props-are-equal.js";
import { jsx } from "react/jsx-runtime";
const defaultPreset = {
  name: "Default",
  params: {
    ...defaultObjectSizing,
    scale: 0.75,
    speed: 1,
    frame: 0,
    contour: 0.5,
    angle: 0,
    noise: 0,
    innerGlow: 0.5,
    outerGlow: 0.5,
    colorBack: "#000000",
    colors: ["#11206a", "#1f3ba2", "#2f63e7", "#6bd7ff", "#ffe679", "#ff991e", "#ff4c00"]
  }
};
const sepiaPreset = {
  name: "Sepia",
  params: {
    ...defaultObjectSizing,
    scale: 0.75,
    speed: 0.5,
    frame: 0,
    contour: 0.5,
    angle: 0,
    noise: 0.75,
    innerGlow: 0.5,
    outerGlow: 0.5,
    colorBack: "#000000",
    colors: ["#997F45", "#ffffff"]
  }
};
const heatmapPresets = [defaultPreset, sepiaPreset];
const Heatmap = memo(function HeatmapImpl({
  // Own props
  speed = defaultPreset.params.speed,
  frame = defaultPreset.params.frame,
  image = "",
  contour = defaultPreset.params.contour,
  angle = defaultPreset.params.angle,
  noise = defaultPreset.params.noise,
  innerGlow = defaultPreset.params.innerGlow,
  outerGlow = defaultPreset.params.outerGlow,
  colorBack = defaultPreset.params.colorBack,
  colors = defaultPreset.params.colors,
  suspendWhenProcessingImage = false,
  // Sizing props
  fit = defaultPreset.params.fit,
  offsetX = defaultPreset.params.offsetX,
  offsetY = defaultPreset.params.offsetY,
  originX = defaultPreset.params.originX,
  originY = defaultPreset.params.originY,
  rotation = defaultPreset.params.rotation,
  scale = defaultPreset.params.scale,
  worldHeight = defaultPreset.params.worldHeight,
  worldWidth = defaultPreset.params.worldWidth,
  ...props
}) {
  const imageUrl = typeof image === "string" ? image : image.src;
  const [processedStateImage, setProcessedStateImage] = useState(transparentPixel);
  let processedImage;
  if (suspendWhenProcessingImage && typeof window !== "undefined") {
    processedImage = suspend(
      () => toProcessedHeatmap(imageUrl).then((result) => URL.createObjectURL(result.blob)),
      [imageUrl, "heatmap"]
    );
  } else {
    processedImage = processedStateImage;
  }
  useLayoutEffect(() => {
    if (suspendWhenProcessingImage) {
      return;
    }
    if (!imageUrl) {
      setProcessedStateImage(transparentPixel);
      return;
    }
    let url;
    let current = true;
    toProcessedHeatmap(imageUrl).then((result) => {
      if (current) {
        url = URL.createObjectURL(result.blob);
        setProcessedStateImage(url);
      }
    });
    return () => {
      current = false;
    };
  }, [imageUrl, suspendWhenProcessingImage]);
  const uniforms = useMemo(
    () => ({
      // Own uniforms
      u_image: processedImage,
      u_contour: contour,
      u_angle: angle,
      u_noise: noise,
      u_innerGlow: innerGlow,
      u_outerGlow: outerGlow,
      u_colorBack: getShaderColorFromString(colorBack),
      u_colors: colors.map(getShaderColorFromString),
      u_colorsCount: colors.length,
      // Sizing uniforms
      u_fit: ShaderFitOptions[fit],
      u_offsetX: offsetX,
      u_offsetY: offsetY,
      u_originX: originX,
      u_originY: originY,
      u_rotation: rotation,
      u_scale: scale,
      u_worldHeight: worldHeight,
      u_worldWidth: worldWidth
    }),
    [
      speed,
      frame,
      contour,
      angle,
      noise,
      innerGlow,
      outerGlow,
      colors,
      colorBack,
      processedImage,
      fit,
      offsetX,
      offsetY,
      originX,
      originY,
      rotation,
      scale,
      worldHeight,
      worldWidth
    ]
  );
  return /* @__PURE__ */ jsx(
    ShaderMount,
    {
      ...props,
      speed,
      frame,
      fragmentShader: heatmapFragmentShader,
      mipmaps: ["u_image"],
      uniforms
    }
  );
}, colorPropsAreEqual);
export {
  Heatmap,
  defaultPreset,
  heatmapPresets,
  sepiaPreset
};
//# sourceMappingURL=heatmap.js.map
