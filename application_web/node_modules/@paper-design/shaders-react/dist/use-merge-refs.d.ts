import * as React from 'react';
/**
 * Merges an array of refs into a single memoized callback ref or `null`.
 * @see https://floating-ui.com/docs/react-utils#usemergerefs
 */
export declare function useMergeRefs<Instance>(refs: Array<React.Ref<Instance> | undefined>): null | React.Ref<Instance>;
