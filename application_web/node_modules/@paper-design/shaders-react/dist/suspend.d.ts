/**
 * A cut down version of suspend-react.
 * When paper-shaders only supports React 19+ we can use the use hook instead.
 */
type Tuple<T = any> = [T] | T[];
type Await<T> = T extends Promise<infer V> ? V : never;
declare const suspend: <Keys extends Tuple<unknown>, Fn extends (...keys: Keys) => Promise<unknown>>(fn: Fn | Promise<unknown>, keys?: Keys) => Await<ReturnType<Fn>>;
export { suspend };
