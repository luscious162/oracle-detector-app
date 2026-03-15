/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

function colorPropsAreEqual(prevProps, nextProps) {
  for (const key in prevProps) {
    if (key === "colors") {
      const prevIsArray = Array.isArray(prevProps.colors);
      const nextIsArray = Array.isArray(nextProps.colors);
      if (!prevIsArray || !nextIsArray) {
        if (Object.is(prevProps.colors, nextProps.colors) === false) {
          return false;
        }
        continue;
      }
      if (prevProps.colors?.length !== nextProps.colors?.length) {
        return false;
      }
      if (!prevProps.colors?.every((color, index) => color === nextProps.colors?.[index])) {
        return false;
      }
      continue;
    }
    if (Object.is(prevProps[key], nextProps[key]) === false) {
      return false;
    }
  }
  return true;
}
export {
  colorPropsAreEqual
};
//# sourceMappingURL=color-props-are-equal.js.map
