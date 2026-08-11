declare module 'occt-import-js' {
  type OcctModuleOptions = {
    locateFile?: (path: string, scriptDirectory?: string) => string
  }

  type OcctModule = {
    ReadStepFile: (buffer: Uint8Array, params: unknown) => unknown
    ReadIgesFile: (buffer: Uint8Array, params: unknown) => unknown
    ReadBrepFile: (buffer: Uint8Array, params: unknown) => unknown
  }

  export default function occtimportjs(options?: OcctModuleOptions): Promise<OcctModule>
}

declare module 'occt-import-js/dist/occt-import-js.wasm?url' {
  const url: string
  export default url
}
