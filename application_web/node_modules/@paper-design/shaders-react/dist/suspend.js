/* * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                    Paper Shaders                    *
 *       https://github.com/paper-design/shaders       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * */

const isPromise = (promise) => typeof promise === "object" && typeof promise.then === "function";
const globalCache = [];
function shallowEqualArrays(arrA, arrB) {
  if (arrA === arrB) return true;
  if (!arrA || !arrB) return false;
  const len = arrA.length;
  if (arrB.length !== len) return false;
  for (let i = 0; i < len; i++) if (arrA[i] !== arrB[i]) return false;
  return true;
}
function query(fn, keys = null) {
  if (keys === null) keys = [fn];
  for (const entry2 of globalCache) {
    if (shallowEqualArrays(keys, entry2.keys)) {
      if (Object.prototype.hasOwnProperty.call(entry2, "error")) throw entry2.error;
      if (Object.prototype.hasOwnProperty.call(entry2, "response")) {
        return entry2.response;
      }
      throw entry2.promise;
    }
  }
  const entry = {
    keys,
    promise: (
      // Execute the promise
      (isPromise(fn) ? fn : fn(...keys)).then((response) => {
        entry.response = response;
      }).catch((error) => entry.error = error)
    )
  };
  globalCache.push(entry);
  throw entry.promise;
}
const suspend = (fn, keys) => query(fn, keys);
export {
  suspend
};
//# sourceMappingURL=suspend.js.map
