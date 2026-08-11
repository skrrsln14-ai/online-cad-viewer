/// <reference lib="webworker" />

import occtimportjs from 'occt-import-js'
import wasmUrl from 'occt-import-js/dist/occt-import-js.wasm?url'
import { DEFAULT_OCCT_TESSELLATION, type OcctTessellationParams } from './occtTessellation'

export type OcctWorkerRequest = {
  id: number
  kind: 'step' | 'iges'
  buffer: ArrayBuffer
  params?: OcctTessellationParams
}

export type OcctWorkerProgress = {
  id: number
  type: 'progress'
  progress: number
  stage: string
}

export type OcctWorkerResult = {
  id: number
  type: 'result'
  result: unknown
}

export type OcctWorkerError = {
  id: number
  type: 'error'
  message: string
}

type OcctModule = {
  ReadStepFile: (buffer: Uint8Array, params: unknown) => unknown
  ReadIgesFile: (buffer: Uint8Array, params: unknown) => unknown
}

let occtReady: Promise<OcctModule> | null = null

function initOcct(): Promise<OcctModule> {
  if (!occtReady) {
    occtReady = occtimportjs({
      locateFile: (path: string) => (path.endsWith('.wasm') ? wasmUrl : path),
    }) as Promise<OcctModule>
  }
  return occtReady
}

self.onmessage = async (event: MessageEvent<OcctWorkerRequest>) => {
  const { id, kind, buffer, params } = event.data
  const tess = params ?? DEFAULT_OCCT_TESSELLATION

  const postProgress = (progress: number, stage: string) => {
    const msg: OcctWorkerProgress = { id, type: 'progress', progress, stage }
    self.postMessage(msg)
  }

  try {
    postProgress(8, 'WASM hazırlanıyor')
    const occt = await initOcct()

    postProgress(22, 'Tessellation başlıyor')
    const fileBuffer = new Uint8Array(buffer)

    // Blocking OCCT call — runs off the UI thread inside this worker
    postProgress(35, kind === 'iges' ? 'IGES parse ediliyor' : 'STEP parse ediliyor')
    const result =
      kind === 'iges'
        ? occt.ReadIgesFile(fileBuffer, tess)
        : occt.ReadStepFile(fileBuffer, tess)

    postProgress(92, 'Mesh verisi aktarılıyor')
    const msg: OcctWorkerResult = { id, type: 'result', result }
    self.postMessage(msg)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'CAD worker hatası'
    const msg: OcctWorkerError = { id, type: 'error', message }
    self.postMessage(msg)
  }
}

export {}
