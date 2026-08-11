/** Fast-first-load tessellation defaults for occt-import-js. */
export type OcctTessellationParams = {
  linearUnit: 'millimeter' | 'centimeter' | 'meter' | 'inch' | 'foot'
  linearDeflectionType: 'bounding_box_ratio' | 'absolute_value'
  /** Coarser = faster. 0.1 bounding-box ratio is a good preview default. */
  linearDeflection: number
  /** Angular deflection in radians; higher = fewer triangles. */
  angularDeflection: number
}

/**
 * Default STEP/IGES tessellation — tuned for load speed.
 * Maps to OCCT linearDeflection / angularDeflection (user-facing “tolerance”).
 */
export const DEFAULT_OCCT_TESSELLATION: OcctTessellationParams = {
  linearUnit: 'millimeter',
  linearDeflectionType: 'bounding_box_ratio',
  linearDeflection: 0.1,
  angularDeflection: 0.1,
}
