/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

function getEmptyPixel() {
  if (typeof window === "undefined") {
    console.warn("Paper Shaders: can\u2019t create an image on the server");
    return void 0;
  }
  const img = new Image();
  img.src = emptyPixel;
  return img;
}
const emptyPixel = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
export {
  getEmptyPixel
};
//# sourceMappingURL=get-empty-pixel.js.map
