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
import wasmUrl from 'occt-import-js/dist/occt-import-js.wasm?url'

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

type OcctImportResult = {
  success?: boolean
  meshes?: OcctMeshData[]
}

export type OcctModule = {
  ReadStepFile: (buffer: Uint8Array, params: unknown) => OcctImportResult
  ReadIgesFile: (buffer: Uint8Array, params: unknown) => OcctImportResult
}

/** Light aluminum / studio CAD default (edges added in CADViewer.wrapLoadedObject). */
const BASE = {
  color: '#d0d7de',
  metalness: 0.2,
  roughness: 0.5,
} as const

let occtReady: Promise<OcctModule> | null = null

/** Initialize OpenCascade WASM (singleton). */
export async function initOcct(): Promise<OcctModule> {
  if (!occtReady) {
    occtReady = (async () => {
      const { default: occtimportjs } = await import('occt-import-js')
      return occtimportjs({
        locateFile: (path: string) => (path.endsWith('.wasm') ? wasmUrl : path),
      }) as Promise<OcctModule>
    })()
  }
  return occtReady
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

/** Center all mesh geometries around the assembly origin (like geometry.center()). */
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

/**
 * Parse STEP/IGES via OpenCascade WASM and return a Three.js object.
 * CAD edge outlines are attached later in CADViewer.wrapLoadedObject.
 */
export async function loadOcctContent(file: File): Promise<Object3D> {
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
  const fileBuffer = new Uint8Array(await file.arrayBuffer())
  const occt = await initOcct()

  const result =
    ext === '.iges' || ext === '.igs'
      ? occt.ReadIgesFile(fileBuffer, null)
      : occt.ReadStepFile(fileBuffer, null)

  if (!result?.meshes?.length) {
    throw new Error('CAD dosyası parse edilemedi veya boş mesh döndü.')
  }

  const group = new Group()
  group.name = file.name

  for (const meshData of result.meshes) {
    group.add(buildPartFromOcct(meshData))
  }

  centerAssembliesAtOrigin(group)
  return group
}
