import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Group,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Vector3,
} from 'three'
import { DEFAULT_OCCT_TESSELLATION, type OcctTessellationParams } from './occtTessellation'
import type {
  OcctWorkerError,
  OcctWorkerProgress,
  OcctWorkerRequest,
  OcctWorkerResult,
} from './occt.worker'

type OcctVec3 = [number, number, number]

type OcctMeshData = {
  name?: string
  color?: OcctVec3
  brep_faces?: Array<{ first: number; last: number; color?: OcctVec3 | null }>
  attributes: {
    position: { array: number[] | Float32Array }
    normal?: { array: number[] | Float32Array }
  }
  index: { array: number[] | Uint32Array }
}

export type OcctImportResult = {
  success?: boolean
  meshes?: OcctMeshData[]
}

export type OcctLoadProgress = {
  progress: number
  stage: string
}

/** Light aluminum / studio CAD default (edges added in CADViewer.wrapLoadedObject). */
const BASE = {
  color: '#d0d7de',
  metalness: 0.18,
  roughness: 0.48,
} as const

let worker: Worker | null = null
let nextRequestId = 1

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./occt.worker.ts', import.meta.url), {
      type: 'module',
    })
  }
  return worker
}

function makeMaterial(color?: OcctVec3 | null): MeshStandardMaterial {
  return new MeshStandardMaterial({
    color: color ? new Color(color[0], color[1], color[2]) : BASE.color,
    metalness: BASE.metalness,
    roughness: BASE.roughness,
  })
}

function buildGeometryFromOcct(meshData: OcctMeshData): {
  geometry: BufferGeometry
  materials: MeshStandardMaterial[]
} {
  const geometry = new BufferGeometry()
  geometry.setAttribute(
    'position',
    new Float32BufferAttribute(meshData.attributes.position.array, 3),
  )

  if (meshData.attributes.normal) {
    geometry.setAttribute(
      'normal',
      new Float32BufferAttribute(meshData.attributes.normal.array, 3),
    )
  } else {
    geometry.computeVertexNormals()
  }

  const index = Uint32Array.from(meshData.index.array)
  geometry.setIndex(new BufferAttribute(index, 1))
  if (meshData.name) geometry.name = meshData.name

  const defaultMaterial = makeMaterial(meshData.color)
  const materials: MeshStandardMaterial[] = [defaultMaterial]
  const faces = meshData.brep_faces ?? []

  if (faces.length > 0) {
    for (const face of faces) {
      materials.push(makeMaterial(face.color ?? meshData.color))
    }

    const triangleCount = index.length / 3
    let triangleIndex = 0
    let faceColorGroupIndex = 0

    while (triangleIndex < triangleCount) {
      let lastIndex: number
      let materialIndex: number
      const face = faces[faceColorGroupIndex]

      if (faceColorGroupIndex >= faces.length || !face) {
        lastIndex = triangleCount
        materialIndex = 0
      } else if (triangleIndex < face.first) {
        lastIndex = face.first
        materialIndex = 0
      } else {
        lastIndex = face.last + 1
        materialIndex = faceColorGroupIndex + 1
        faceColorGroupIndex += 1
      }

      geometry.addGroup(triangleIndex * 3, (lastIndex - triangleIndex) * 3, materialIndex)
      triangleIndex = lastIndex
    }
  }

  return { geometry, materials }
}

function centerAssembliesAtOrigin(root: Object3D) {
  root.updateMatrixWorld(true)
  const box = new Box3().setFromObject(root)
  if (box.isEmpty()) return

  const center = box.getCenter(new Vector3())
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const mesh = child as Mesh
    mesh.geometry.translate(-center.x, -center.y, -center.z)
  })
}

function buildPartFromOcct(meshData: OcctMeshData): Group {
  const { geometry, materials } = buildGeometryFromOcct(meshData)
  const mesh = new Mesh(geometry, materials.length > 1 ? materials : materials[0])
  if (meshData.name) mesh.name = meshData.name
  mesh.castShadow = true
  mesh.receiveShadow = true

  const part = new Group()
  part.name = meshData.name ?? 'occt-part'
  part.add(mesh)
  return part
}

export function isOcctCadFile(filename: string): boolean {
  const i = filename.lastIndexOf('.')
  const ext = i >= 0 ? filename.slice(i).toLowerCase() : ''
  return ext === '.step' || ext === '.stp' || ext === '.iges' || ext === '.igs'
}

function buildObjectFromResult(result: OcctImportResult, name: string): Object3D {
  if (!result?.meshes?.length) {
    throw new Error('CAD dosyası parse edilemedi veya boş mesh döndü.')
  }

  const group = new Group()
  group.name = name

  for (const meshData of result.meshes) {
    group.add(buildPartFromOcct(meshData))
  }

  centerAssembliesAtOrigin(group)
  return group
}

/**
 * Parse STEP/IGES on a Web Worker (WASM + tessellation off the UI thread).
 */
export function parseOcctInWorker(
  buffer: ArrayBuffer,
  kind: 'step' | 'iges',
  onProgress?: (info: OcctLoadProgress) => void,
  params: OcctTessellationParams = DEFAULT_OCCT_TESSELLATION,
): Promise<OcctImportResult> {
  const w = getWorker()
  const id = nextRequestId++

  return new Promise((resolve, reject) => {
    let softProgress = 35
    let softTimer: ReturnType<typeof setInterval> | null = null

    const stopSoftProgress = () => {
      if (softTimer) {
        clearInterval(softTimer)
        softTimer = null
      }
    }

    const startSoftProgress = () => {
      stopSoftProgress()
      softTimer = setInterval(() => {
        // Creep toward ~88% while OCCT blocks inside the worker
        softProgress = Math.min(88, softProgress + Math.max(0.35, (88 - softProgress) * 0.04))
        onProgress?.({
          progress: softProgress,
          stage: kind === 'iges' ? 'IGES tessellation…' : 'STEP tessellation…',
        })
      }, 250)
    }

    const handleMessage = (
      event: MessageEvent<OcctWorkerProgress | OcctWorkerResult | OcctWorkerError>,
    ) => {
      const data = event.data
      if (data.id !== id) return

      if (data.type === 'progress') {
        if (data.progress >= 35 && data.progress < 90) {
          softProgress = Math.max(softProgress, data.progress)
          startSoftProgress()
        } else {
          stopSoftProgress()
          onProgress?.({ progress: data.progress, stage: data.stage })
        }
        return
      }

      stopSoftProgress()
      w.removeEventListener('message', handleMessage)
      w.removeEventListener('error', handleError)

      if (data.type === 'error') {
        reject(new Error(data.message))
        return
      }

      resolve(data.result as OcctImportResult)
    }

    const handleError = (err: ErrorEvent) => {
      stopSoftProgress()
      w.removeEventListener('message', handleMessage)
      w.removeEventListener('error', handleError)
      reject(err.error instanceof Error ? err.error : new Error(err.message || 'Worker hatası'))
    }

    w.addEventListener('message', handleMessage)
    w.addEventListener('error', handleError)

    const request: OcctWorkerRequest = { id, kind, buffer, params }
    // Transfer ownership of the ArrayBuffer to the worker (zero-copy)
    w.postMessage(request, [buffer])
  })
}

/**
 * Parse STEP/IGES via OpenCascade WASM (worker) and build a Three.js object.
 * CAD edge outlines are attached later in CADViewer.wrapLoadedObject.
 */
export async function loadOcctContent(
  file: File,
  onProgress?: (info: OcctLoadProgress) => void,
  params: OcctTessellationParams = DEFAULT_OCCT_TESSELLATION,
): Promise<Object3D> {
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
  const kind: 'step' | 'iges' = ext === '.iges' || ext === '.igs' ? 'iges' : 'step'

  onProgress?.({ progress: 3, stage: 'Dosya okunuyor' })
  const buffer = await file.arrayBuffer()

  onProgress?.({ progress: 6, stage: 'Arka plan işçisine aktarılıyor' })
  const result = await parseOcctInWorker(buffer, kind, onProgress, params)

  onProgress?.({ progress: 94, stage: 'Three.js mesh oluşturuluyor' })
  const object = buildObjectFromResult(result, file.name)

  onProgress?.({ progress: 100, stage: 'Tamamlandı' })
  return object
}

export { DEFAULT_OCCT_TESSELLATION }
export type { OcctTessellationParams }
