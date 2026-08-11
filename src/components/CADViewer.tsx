import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type MutableRefObject,
} from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html, Line, OrbitControls } from '@react-three/drei'
import {
  AlwaysStencilFunc,
  BackSide,
  Box3,
  BufferAttribute,
  BufferGeometry,
  DecrementWrapStencilOp,
  EdgesGeometry,
  FrontSide,
  Group,
  IncrementWrapStencilOp,
  LineBasicMaterial,
  LineSegments,
  Material,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  NotEqualStencilFunc,
  Object3D,
  Plane,
  PlaneGeometry,
  PlaneHelper,
  Quaternion,
  Raycaster,
  RepeatWrapping,
  ReplaceStencilOp,
  Sphere,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  Vector2,
  Vector3,
  DirectionalLight,
} from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { loadOcctContent } from '../lib/loadOcctModel'
import ViewCube, { CameraOrientationPublisher } from './ViewCube'
import './CADViewer.css'

type ViewMode = 'solid' | 'wireframe' | 'translucent'
type ViewPreset = 'iso' | 'top' | 'front' | 'right'
type Axis = 'x' | 'y' | 'z'
type TextureTarget = 'model' | 'ground'

const GROUND_TEXTURE_REPEAT = 10
const GROUND_BASE_COLOR = '#2a2a2a'

type ModelStats = {
  triangles: number
  size: { x: number; y: number; z: number }
}

type LoadedModel = {
  root: Group
  name: string
  stats: ModelStats
}

type OrbitControlsLike = {
  target: Vector3
  update: () => void
  enabled: boolean
}

const BASE_MATERIAL = {
  color: '#c0c6cc',
  metalness: 0.2,
  roughness: 0.5,
} as const

const EDGE_THRESHOLD_DEG = 35
const EDGE_COLOR = '#1a1a1a'
const DEFAULT_MODEL_COLOR = '#c0c6cc'

type Measurement = {
  id: string
  a: [number, number, number]
  b: [number, number, number]
}

type ClipAxisState = {
  enabled: boolean
  value: number
  inverted: boolean
}

type ClipBounds = {
  min: { x: number; y: number; z: number }
  max: { x: number; y: number; z: number }
}

const DEFAULT_CLIP_AXIS: ClipAxisState = {
  enabled: false,
  value: 0,
  inverted: false,
}

const CLIP_HELPER_COLORS = {
  x: 0xff5555,
  y: 0x55ff88,
  z: 0x5599ff,
} as const

type AssemblyPart = {
  id: string
  name: string
  visible: boolean
}

function buildAssemblyParts(root: Object3D): AssemblyPart[] {
  const parts: AssemblyPart[] = []
  let index = 0

  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    if (child.name === 'selection-highlight') return

    index += 1
    const mesh = child as Mesh
    const raw = mesh.name?.trim()
    const name = raw && raw !== 'cad-edges' ? raw : `Parça ${index}`

    parts.push({
      id: mesh.uuid,
      name,
      visible: mesh.visible,
    })
  })

  return parts
}

function findMeshById(root: Object3D, id: string): Mesh | null {
  let found: Mesh | null = null
  root.traverse((child) => {
    if ((child as Mesh).isMesh && child.uuid === id) {
      found = child as Mesh
    }
  })
  return found
}

function isolateMesh(root: Object3D, id: string) {
  root.traverse((child) => {
    if ((child as Mesh).isMesh && child.name !== 'selection-highlight') {
      child.visible = child.uuid === id
    }
  })
}

function showAllMeshes(root: Object3D) {
  root.traverse((child) => {
    if ((child as Mesh).isMesh && child.name !== 'selection-highlight') {
      child.visible = true
    }
  })
}

const MODEL_EXTENSIONS = ['.stl', '.obj', '.step', '.stp', '.iges', '.igs'] as const
const TEXTURE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'] as const
const OCCT_EXTENSIONS = ['.step', '.stp', '.iges', '.igs'] as const

const VIEW_DIRECTIONS: Record<ViewPreset, Vector3> = {
  iso: new Vector3(1, 0.85, 1).normalize(),
  top: new Vector3(0, 1, 0.0001).normalize(),
  front: new Vector3(0, 0.12, 1).normalize(),
  right: new Vector3(1, 0.12, 0).normalize(),
}

function getExtension(filename: string): string {
  const i = filename.lastIndexOf('.')
  return i >= 0 ? filename.slice(i).toLowerCase() : ''
}

function isModelFile(file: File): boolean {
  return (MODEL_EXTENSIONS as readonly string[]).includes(getExtension(file.name))
}

function isTextureFile(file: File): boolean {
  const ext = getExtension(file.name)
  if ((TEXTURE_EXTENSIONS as readonly string[]).includes(ext)) return true
  return file.type.startsWith('image/')
}

function createMaterial(): MeshStandardMaterial {
  return new MeshStandardMaterial({ ...BASE_MATERIAL })
}

function collectRenderableMeshes(root: Object3D): Mesh[] {
  const meshes: Mesh[] = []
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    if (child.name === 'selection-highlight') return
    if (child.name.startsWith('clip-stencil')) return
    if (child.name.startsWith('clip-cap')) return
    meshes.push(child as Mesh)
  })
  return meshes
}

function disposeObject3DTree(root: Object3D) {
  root.traverse((child) => {
    const mesh = child as Mesh
    if (mesh.isMesh) {
      // Shared geometries from model meshes must NOT be disposed here
      if (mesh.name.startsWith('clip-cap')) {
        mesh.geometry?.dispose()
      }
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
      materials.forEach((mat) => mat?.dispose())
    }
  })
}

function createPlaneStencilGroup(
  geometry: BufferGeometry,
  plane: Plane,
  renderOrder: number,
): Group {
  const group = new Group()
  group.name = 'clip-stencil'

  const baseMat = new MeshBasicMaterial()
  baseMat.depthWrite = false
  baseMat.depthTest = false
  baseMat.colorWrite = false
  baseMat.stencilWrite = true
  baseMat.stencilFunc = AlwaysStencilFunc

  const matBack = baseMat.clone()
  matBack.side = BackSide
  matBack.clippingPlanes = [plane]
  matBack.stencilFail = IncrementWrapStencilOp
  matBack.stencilZFail = IncrementWrapStencilOp
  matBack.stencilZPass = IncrementWrapStencilOp

  const matFront = baseMat.clone()
  matFront.side = FrontSide
  matFront.clippingPlanes = [plane]
  matFront.stencilFail = DecrementWrapStencilOp
  matFront.stencilZFail = DecrementWrapStencilOp
  matFront.stencilZPass = DecrementWrapStencilOp

  const backMesh = new Mesh(geometry, matBack)
  backMesh.name = 'clip-stencil-back'
  backMesh.renderOrder = renderOrder
  backMesh.raycast = () => undefined
  group.add(backMesh)

  const frontMesh = new Mesh(geometry, matFront)
  frontMesh.name = 'clip-stencil-front'
  frontMesh.renderOrder = renderOrder
  frontMesh.raycast = () => undefined
  group.add(frontMesh)

  return group
}

function clearClipStencilChildren(root: Object3D) {
  const toRemove: Object3D[] = []
  root.traverse((child) => {
    if (child.name === 'clip-stencil') toRemove.push(child)
  })
  for (const node of toRemove) {
    node.parent?.remove(node)
    disposeObject3DTree(node)
  }
}

function applyClippingToObject(root: Object3D | null, planes: Plane[]) {
  if (!root) return
  const list = planes.length > 0 ? planes : []

  root.traverse((child) => {
    if (child.name.startsWith('clip-stencil') || child.name.startsWith('clip-cap')) return

    const mesh = child as Mesh
    if (mesh.isMesh) {
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
      materials.forEach((mat) => {
        if (!mat) return
        mat.clippingPlanes = list
        mat.clipShadows = list.length > 0
        mat.needsUpdate = true
      })
      mesh.renderOrder = list.length > 0 ? 6 : 0
    }

    const lines = child as LineSegments
    if (lines.isLineSegments) {
      const materials = Array.isArray(lines.material) ? lines.material : [lines.material]
      materials.forEach((mat) => {
        if (!mat) return
        mat.clippingPlanes = list
        mat.needsUpdate = true
      })
    }
  })
}

function updateClipPlane(plane: Plane, axis: Axis, value: number, inverted: boolean) {
  if (axis === 'x') {
    plane.normal.set(inverted ? 1 : -1, 0, 0)
    plane.constant = inverted ? -value : value
  } else if (axis === 'y') {
    plane.normal.set(0, inverted ? 1 : -1, 0)
    plane.constant = inverted ? -value : value
  } else {
    plane.normal.set(0, 0, inverted ? 1 : -1)
    plane.constant = inverted ? -value : value
  }
  plane.normalize()
}

function getObjectClipBounds(root: Object3D | null): ClipBounds {
  if (!root) {
    return { min: { x: -10, y: 0, z: -10 }, max: { x: 10, y: 10, z: 10 } }
  }
  const box = new Box3().setFromObject(root)
  if (box.isEmpty()) {
    return { min: { x: -10, y: 0, z: -10 }, max: { x: 10, y: 10, z: 10 } }
  }
  const pad = 0.001
  return {
    min: { x: box.min.x - pad, y: box.min.y - pad, z: box.min.z - pad },
    max: { x: box.max.x + pad, y: box.max.y + pad, z: box.max.z + pad },
  }
}

function ClippingController({
  model,
  clipMasterEnabled,
  clipX,
  clipY,
  clipZ,
  draggingAxis,
  showHelpersAlways,
  solidCapEnabled,
  capColor,
  capsRevision,
}: {
  model: LoadedModel | null
  clipMasterEnabled: boolean
  clipX: ClipAxisState
  clipY: ClipAxisState
  clipZ: ClipAxisState
  draggingAxis: Axis | null
  showHelpersAlways: boolean
  solidCapEnabled: boolean
  capColor: string
  capsRevision: number
}) {
  const { gl } = useThree()
  const planes = useMemo(
    () => ({
      x: new Plane(new Vector3(-1, 0, 0), 0),
      y: new Plane(new Vector3(0, -1, 0), 0),
      z: new Plane(new Vector3(0, 0, -1), 0),
    }),
    [],
  )

  const helpers = useMemo(() => {
    const size = 40
    return {
      x: new PlaneHelper(planes.x, size, CLIP_HELPER_COLORS.x),
      y: new PlaneHelper(planes.y, size, CLIP_HELPER_COLORS.y),
      z: new PlaneHelper(planes.z, size, CLIP_HELPER_COLORS.z),
    }
  }, [planes])

  const capsGroupRef = useRef<Group>(null)
  const capMeshesRef = useRef<Partial<Record<Axis, Mesh>>>({})

  useEffect(() => {
    gl.localClippingEnabled = true
  }, [gl])

  // Keep plane equations + model clippingPlanes in sync (cheap; runs on slider move)
  useEffect(() => {
    updateClipPlane(planes.x, 'x', clipX.value, clipX.inverted)
    updateClipPlane(planes.y, 'y', clipY.value, clipY.inverted)
    updateClipPlane(planes.z, 'z', clipZ.value, clipZ.inverted)

    const activePlanes: Plane[] = []
    if (clipMasterEnabled) {
      if (clipX.enabled) activePlanes.push(planes.x)
      if (clipY.enabled) activePlanes.push(planes.y)
      if (clipZ.enabled) activePlanes.push(planes.z)
    }
    applyClippingToObject(model?.root ?? null, activePlanes)

    const diag = model
      ? Math.max(model.stats.size.x, model.stats.size.y, model.stats.size.z, 10) * 1.4
      : 40
    helpers.x.size = diag
    helpers.y.size = diag
    helpers.z.size = diag
    helpers.x.updateMatrixWorld(true)
    helpers.y.updateMatrixWorld(true)
    helpers.z.updateMatrixWorld(true)
  }, [clipMasterEnabled, clipX, clipY, clipZ, model, planes, helpers])

  // Rebuild stencil groups + cap meshes only when structure/color changes
  useEffect(() => {
    updateClipPlane(planes.x, 'x', clipX.value, clipX.inverted)
    updateClipPlane(planes.y, 'y', clipY.value, clipY.inverted)
    updateClipPlane(planes.z, 'z', clipZ.value, clipZ.inverted)

    const activePlanes: Plane[] = []
    const activeAxes: Axis[] = []
    if (clipMasterEnabled) {
      if (clipX.enabled) {
        activePlanes.push(planes.x)
        activeAxes.push('x')
      }
      if (clipY.enabled) {
        activePlanes.push(planes.y)
        activeAxes.push('y')
      }
      if (clipZ.enabled) {
        activePlanes.push(planes.z)
        activeAxes.push('z')
      }
    }

    if (model?.root) clearClipStencilChildren(model.root)

    const capsRoot = capsGroupRef.current
    if (capsRoot) {
      while (capsRoot.children.length) {
        const child = capsRoot.children[0]
        capsRoot.remove(child)
        disposeObject3DTree(child)
      }
    }
    capMeshesRef.current = {}

    if (
      !model?.root ||
      !clipMasterEnabled ||
      !solidCapEnabled ||
      activeAxes.length === 0 ||
      !capsRoot
    ) {
      return
    }

    const meshes = collectRenderableMeshes(model.root).filter((m) => m.visible)
    const planeByAxis: Record<Axis, Plane> = {
      x: planes.x,
      y: planes.y,
      z: planes.z,
    }
    const diag = Math.max(model.stats.size.x, model.stats.size.y, model.stats.size.z, 10) * 1.4

    activeAxes.forEach((axis, index) => {
      const plane = planeByAxis[axis]
      const renderOrder = index + 1

      for (const mesh of meshes) {
        mesh.add(createPlaneStencilGroup(mesh.geometry, plane, renderOrder))
      }

      const otherPlanes = activePlanes.filter((p) => p !== plane)
      const planeMat = new MeshStandardMaterial({
        color: capColor,
        metalness: 0.05,
        roughness: 0.85,
        clippingPlanes: otherPlanes,
        clipShadows: true,
        shadowSide: BackSide,
        side: FrontSide,
        stencilWrite: true,
        stencilRef: 0,
        stencilFunc: NotEqualStencilFunc,
        stencilFail: ReplaceStencilOp,
        stencilZFail: ReplaceStencilOp,
        stencilZPass: ReplaceStencilOp,
      })

      const capSize = Math.max(diag * 2.2, 20)
      const capMesh = new Mesh(new PlaneGeometry(capSize, capSize), planeMat)
      capMesh.name = `clip-cap-${axis}`
      capMesh.renderOrder = renderOrder + 0.1
      capMesh.raycast = () => undefined
      capMesh.onAfterRender = (renderer) => {
        renderer.clearStencil()
      }
      capsRoot.add(capMesh)
      capMeshesRef.current[axis] = capMesh
    })

    return () => {
      if (model?.root) clearClipStencilChildren(model.root)
    }
  }, [
    clipMasterEnabled,
    clipX.enabled,
    clipX.inverted,
    clipY.enabled,
    clipY.inverted,
    clipZ.enabled,
    clipZ.inverted,
    model,
    planes,
    solidCapEnabled,
    capsRevision,
    // capColor updates via dedicated effect — avoid full stencil rebuild
  ])

  useEffect(() => {
    Object.values(capMeshesRef.current).forEach((cap) => {
      if (!cap) return
      const mat = cap.material as MeshStandardMaterial
      mat.color.set(capColor)
      mat.needsUpdate = true
    })
  }, [capColor])

  useFrame(() => {
    const planeByAxis: Record<Axis, Plane> = {
      x: planes.x,
      y: planes.y,
      z: planes.z,
    }
    ;(['x', 'y', 'z'] as Axis[]).forEach((axis) => {
      const cap = capMeshesRef.current[axis]
      const plane = planeByAxis[axis]
      if (!cap) return
      plane.coplanarPoint(cap.position)
      cap.lookAt(
        cap.position.x - plane.normal.x,
        cap.position.y - plane.normal.y,
        cap.position.z - plane.normal.z,
      )
    })
  })

  const showX =
    clipMasterEnabled &&
    clipX.enabled &&
    (showHelpersAlways || draggingAxis === 'x')
  const showY =
    clipMasterEnabled &&
    clipY.enabled &&
    (showHelpersAlways || draggingAxis === 'y')
  const showZ =
    clipMasterEnabled &&
    clipZ.enabled &&
    (showHelpersAlways || draggingAxis === 'z')

  return (
    <>
      <group ref={capsGroupRef} />
      {showX && <primitive object={helpers.x} />}
      {showY && <primitive object={helpers.y} />}
      {showZ && <primitive object={helpers.z} />}
    </>
  )
}


function attachCadEdges(mesh: Mesh) {
  if (mesh.getObjectByName('cad-edges')) return
  const lines = new LineSegments(
    new EdgesGeometry(mesh.geometry, EDGE_THRESHOLD_DEG),
    new LineBasicMaterial({
      color: EDGE_COLOR,
      transparent: true,
      opacity: 0.88,
      depthTest: true,
    }),
  )
  lines.name = 'cad-edges'
  lines.renderOrder = 2
  mesh.add(lines)
}

/** Add dark CAD edge outlines to every mesh (holes, fillets, transitions). */
function addCadEdgeOutlines(root: Object3D) {
  const meshes: Mesh[] = []
  root.traverse((child) => {
    if ((child as Mesh).isMesh) meshes.push(child as Mesh)
  })
  for (const mesh of meshes) attachCadEdges(mesh)
}

function applyBaseMaterial(root: Object3D) {
  root.traverse((child) => {
    if ((child as Mesh).isMesh) {
      const mesh = child as Mesh
      mesh.material = createMaterial()
      mesh.castShadow = true
      mesh.receiveShadow = true
    }
  })
}

function countTriangles(root: Object3D): number {
  let triangles = 0
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const geometry = (child as Mesh).geometry as BufferGeometry
    if (!geometry) return
    if (geometry.index) {
      triangles += geometry.index.count / 3
    } else if (geometry.attributes.position) {
      triangles += geometry.attributes.position.count / 3
    }
  })
  return Math.round(triangles)
}

function getWorldSize(root: Object3D): { x: number; y: number; z: number } {
  root.updateMatrixWorld(true)
  const size = new Box3().setFromObject(root).getSize(new Vector3())
  return { x: size.x, y: size.y, z: size.z }
}

/** Center on XZ origin and seat the bounding-box floor on Y = 0. */
function alignToGround(root: Object3D) {
  root.updateMatrixWorld(true)
  const box = new Box3().setFromObject(root)
  if (box.isEmpty()) return
  const center = box.getCenter(new Vector3())
  root.position.x -= center.x
  root.position.z -= center.z
  root.position.y -= box.min.y
  root.updateMatrixWorld(true)
}

function computeStats(root: Object3D): ModelStats {
  return {
    triangles: countTriangles(root),
    size: getWorldSize(root),
  }
}

/**
 * CAD files are often Z-up. Three.js is Y-up — apply -90° on X,
 * then seat the model on the grid plane.
 */
function wrapLoadedObject(object: Object3D): Group {
  const content = new Group()
  content.add(object)
  content.rotation.x = -Math.PI / 2

  const root = new Group()
  root.add(content)
  alignToGround(root)
  addCadEdgeOutlines(root)
  return root
}

function createStlContent(buffer: ArrayBuffer): Object3D {
  const geometry = new STLLoader().parse(buffer) as BufferGeometry
  geometry.computeVertexNormals()
  return new Mesh(geometry, createMaterial())
}

function createObjContent(text: string): Object3D {
  const group = new OBJLoader().parse(text) as Group
  applyBaseMaterial(group)
  return group
}

async function loadModelFromFile(file: File): Promise<LoadedModel> {
  const ext = getExtension(file.name)
  let content: Object3D

  if (ext === '.stl') {
    content = createStlContent(await file.arrayBuffer())
  } else if (ext === '.obj') {
    content = createObjContent(await file.text())
  } else if ((OCCT_EXTENSIONS as readonly string[]).includes(ext)) {
    content = await loadOcctContent(file)
  } else {
    throw new Error(`Desteklenmeyen dosya türü: ${ext || 'bilinmiyor'}`)
  }

  const root = wrapLoadedObject(content)
  return { root, name: file.name, stats: computeStats(root) }
}

function disposeObject(root: Object3D) {
  const disposedTextures = new Set<Texture>()

  root.traverse((child) => {
    if ((child as LineSegments).isLineSegments) {
      const lines = child as LineSegments
      lines.geometry?.dispose()
      const lineMats = Array.isArray(lines.material) ? lines.material : [lines.material]
      lineMats.forEach((mat) => {
        if (mat !== null && typeof mat === 'object' && 'dispose' in mat) {
          ;(mat as Material).dispose()
        }
      })
      return
    }

    if (!(child as Mesh).isMesh) return
    const mesh = child as Mesh
    mesh.geometry?.dispose()
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    materials.forEach((mat: Material) => {
      if (mat instanceof MeshStandardMaterial && mat.map) {
        if (!disposedTextures.has(mat.map)) {
          disposedTextures.add(mat.map)
          mat.map.dispose()
        }
        mat.map = null
      }
      mat.dispose()
    })
  })
}

/** STL and some OBJs lack UVs — generate a simple box projection so textures can show. */
function ensureBoxUVs(root: Object3D) {
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const geometry = (child as Mesh).geometry as BufferGeometry
    if (!geometry || geometry.getAttribute('uv')) return

    geometry.computeBoundingBox()
    geometry.computeVertexNormals()
    const bbox = geometry.boundingBox
    if (!bbox) return

    const size = bbox.getSize(new Vector3())
    const sx = Math.max(size.x, 1e-6)
    const sy = Math.max(size.y, 1e-6)
    const sz = Math.max(size.z, 1e-6)
    const pos = geometry.attributes.position
    const nrm = geometry.attributes.normal
    const uvs = new Float32Array(pos.count * 2)

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i)
      const y = pos.getY(i)
      const z = pos.getZ(i)
      const nx = Math.abs(nrm.getX(i))
      const ny = Math.abs(nrm.getY(i))
      const nz = Math.abs(nrm.getZ(i))

      let u = 0
      let v = 0
      if (nx >= ny && nx >= nz) {
        u = (z - bbox.min.z) / sz
        v = (y - bbox.min.y) / sy
      } else if (ny >= nx && ny >= nz) {
        u = (x - bbox.min.x) / sx
        v = (z - bbox.min.z) / sz
      } else {
        u = (x - bbox.min.x) / sx
        v = (y - bbox.min.y) / sy
      }
      uvs[i * 2] = u
      uvs[i * 2 + 1] = v
    }

    geometry.setAttribute('uv', new BufferAttribute(uvs, 2))
  })
}

function loadTextureFromFile(file: File): Promise<{ texture: Texture; url: string }> {
  const url = URL.createObjectURL(file)
  const loader = new TextureLoader()

  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (texture) => {
        texture.colorSpace = SRGBColorSpace
        texture.flipY = true
        texture.needsUpdate = true
        resolve({ texture, url })
      },
      undefined,
      (err) => {
        URL.revokeObjectURL(url)
        reject(err instanceof Error ? err : new Error('Kaplama yüklenemedi.'))
      },
    )
  })
}

function applyTextureToModel(root: Object3D, texture: Texture) {
  ensureBoxUVs(root)

  const disposed = new Set<Texture>()

  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const mesh = child as Mesh
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]

    materials.forEach((mat) => {
      if (!(mat instanceof MeshStandardMaterial)) return
      if (mat.map && !disposed.has(mat.map)) {
        disposed.add(mat.map)
        mat.map.dispose()
      }
      mat.map = texture
      mat.color.set('#ffffff')
      mat.metalness = Math.min(mat.metalness, 0.25)
      mat.roughness = Math.max(mat.roughness, 0.45)
      mat.needsUpdate = true
    })
  })
}

function applyModelColor(root: Object3D, color: string) {
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const mesh = child as Mesh
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    materials.forEach((mat) => {
      if (!(mat instanceof MeshStandardMaterial)) return
      mat.color.set(color)
      mat.needsUpdate = true
    })
  })
}

function clearTextureFromModel(root: Object3D, restoreColor: string = BASE_MATERIAL.color) {
  const disposed = new Set<Texture>()

  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const mesh = child as Mesh
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    materials.forEach((mat) => {
      if (!(mat instanceof MeshStandardMaterial)) return
      if (mat.map) {
        if (!disposed.has(mat.map)) {
          disposed.add(mat.map)
          mat.map.dispose()
        }
        mat.map = null
      }
      mat.color.set(restoreColor)
      mat.metalness = BASE_MATERIAL.metalness
      mat.roughness = BASE_MATERIAL.roughness
      mat.needsUpdate = true
    })
  })
}

function prepareGroundTexture(texture: Texture): Texture {
  texture.wrapS = RepeatWrapping
  texture.wrapT = RepeatWrapping
  texture.repeat.set(GROUND_TEXTURE_REPEAT, GROUND_TEXTURE_REPEAT)
  texture.needsUpdate = true
  return texture
}

function applyViewMode(root: Object3D, mode: ViewMode) {
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const mesh = child as Mesh
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    materials.forEach((mat) => {
      if (!(mat instanceof MeshStandardMaterial)) return
      mat.wireframe = mode === 'wireframe'
      mat.transparent = mode === 'translucent'
      mat.opacity = mode === 'translucent' ? 0.42 : 1
      mat.depthWrite = mode !== 'translucent'
      if (!mat.map) {
        mat.metalness = mode === 'wireframe' ? 0.2 : BASE_MATERIAL.metalness
        mat.roughness = mode === 'wireframe' ? 0.6 : BASE_MATERIAL.roughness
      }
      mat.needsUpdate = true
    })
  })
}

function rotateModelAxis(model: LoadedModel, axis: Axis): LoadedModel {
  const content = model.root.children[0]
  if (!content) return model

  const angle = Math.PI / 2
  if (axis === 'x') content.rotation.x += angle
  if (axis === 'y') content.rotation.y += angle
  if (axis === 'z') content.rotation.z += angle

  model.root.position.set(0, 0, 0)
  alignToGround(model.root)

  return {
    ...model,
    stats: computeStats(model.root),
  }
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

function CameraDirector({
  target,
  preset,
  requestId,
  focusObject,
  focusRequestId,
  viewDirRef,
  viewDirRequestId,
}: {
  target: Object3D | null
  preset: ViewPreset
  requestId: number
  focusObject?: Object3D | null
  focusRequestId?: number
  viewDirRef?: MutableRefObject<Vector3>
  viewDirRequestId?: number
}) {
  const camera = useThree((state) => state.camera)
  const controls = useThree((state) => state.controls) as OrbitControlsLike | null
  const anim = useRef<{
    t: number
    duration: number
    fromPos: Vector3
    toPos: Vector3
    fromTarget: Vector3
    toTarget: Vector3
  } | null>(null)

  const beginOrbitTo = (direction: Vector3, lookTarget: Vector3, distance: number, duration: number) => {
    const dir = direction.clone().normalize()
    if (Math.abs(dir.y) > 0.999) {
      dir.x += 0.0001
      dir.normalize()
    }
    const toPos = lookTarget.clone().add(dir.multiplyScalar(distance))

    camera.near = Math.max(distance / 100, 0.01)
    camera.far = Math.max(distance * 50, 100)
    camera.updateProjectionMatrix()

    anim.current = {
      t: 0,
      duration,
      fromPos: camera.position.clone(),
      toPos,
      fromTarget: controls?.target.clone() ?? lookTarget.clone(),
      toTarget: lookTarget.clone(),
    }
    if (controls) controls.enabled = false
  }

  useEffect(() => {
    if (!focusObject || !focusRequestId) return

    const box = new Box3().setFromObject(focusObject)
    if (box.isEmpty()) return

    const center = box.getCenter(new Vector3())
    const radius = Math.max(box.getBoundingSphere(new Sphere()).radius, 0.01)
    const fov = 'fov' in camera ? ((camera.fov as number) * Math.PI) / 180 : Math.PI / 4
    const distance = (radius / Math.sin(fov / 2)) * 1.35
    let direction = camera.position.clone().sub(center)
    if (direction.lengthSq() < 1e-6) direction = VIEW_DIRECTIONS.iso.clone()
    else direction.normalize()
    beginOrbitTo(direction, center, distance, 0.5)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusObject, focusRequestId, camera, controls])

  useEffect(() => {
    if (!target || requestId === 0) return

    const box = new Box3().setFromObject(target)
    if (box.isEmpty()) return

    const center = box.getCenter(new Vector3())
    const radius = Math.max(box.getBoundingSphere(new Sphere()).radius, 0.01)
    const fov = 'fov' in camera ? ((camera.fov as number) * Math.PI) / 180 : Math.PI / 4
    const distance = (radius / Math.sin(fov / 2)) * 1.2
    beginOrbitTo(VIEW_DIRECTIONS[preset], center, distance, 0.55)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, preset, requestId, camera, controls])

  useEffect(() => {
    if (!viewDirRef || !viewDirRequestId) return

    const lookTarget = new Vector3()
    let distance = camera.position.distanceTo(controls?.target ?? lookTarget)

    if (target) {
      const box = new Box3().setFromObject(target)
      if (!box.isEmpty()) {
        box.getCenter(lookTarget)
        const radius = Math.max(box.getBoundingSphere(new Sphere()).radius, 0.01)
        const fov = 'fov' in camera ? ((camera.fov as number) * Math.PI) / 180 : Math.PI / 4
        distance = (radius / Math.sin(fov / 2)) * 1.2
      } else if (controls) {
        lookTarget.copy(controls.target)
      }
    } else if (controls) {
      lookTarget.copy(controls.target)
      if (distance < 0.01) distance = 5
    } else {
      return
    }

    if (distance < 0.01) distance = 5
    beginOrbitTo(viewDirRef.current, lookTarget, distance, 0.5)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewDirRequestId, viewDirRef, target, camera, controls])

  useFrame((_, delta) => {
    const current = anim.current
    if (!current) return

    current.t = Math.min(1, current.t + delta / current.duration)
    const k = easeInOutCubic(current.t)

    camera.position.lerpVectors(current.fromPos, current.toPos, k)
    if (controls) {
      controls.target.lerpVectors(current.fromTarget, current.toTarget, k)
      controls.update()
    } else {
      camera.lookAt(current.toTarget)
    }

    if (current.t >= 1) {
      anim.current = null
      if (controls) controls.enabled = true
    }
  })

  return null
}

function GroundPlane({
  size,
  texture,
}: {
  size: number
  texture: Texture | null
}) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.002, 0]} receiveShadow>
      <planeGeometry args={[size, size]} />
      <meshStandardMaterial
        key={texture?.uuid ?? 'ground-base'}
        map={texture}
        color={texture ? '#ffffff' : GROUND_BASE_COLOR}
        metalness={0.05}
        roughness={0.92}
      />
    </mesh>
  )
}

function KeyLightFollowCamera({ intensity = 2.5 }: { intensity?: number }) {
  const lightRef = useRef<DirectionalLight>(null)
  const camera = useThree((state) => state.camera)
  const controls = useThree((state) => state.controls) as OrbitControlsLike | null
  const lookTarget = useRef(new Vector3())
  const toCamera = useRef(new Vector3())
  const right = useRef(new Vector3())
  const camUp = useRef(new Vector3())

  useFrame(() => {
    const light = lightRef.current
    if (!light) return

    if (controls?.target) {
      lookTarget.current.copy(controls.target)
    } else {
      lookTarget.current.set(0, 0, 0)
    }

    toCamera.current.copy(camera.position).sub(lookTarget.current)
    const distance = Math.max(toCamera.current.length(), 1)

    right.current.crossVectors(toCamera.current, new Vector3(0, 1, 0))
    if (right.current.lengthSq() < 1e-6) {
      right.current.set(1, 0, 0)
    } else {
      right.current.normalize()
    }
    camUp.current.crossVectors(right.current, toCamera.current).normalize()
    toCamera.current.normalize()

    // Studio key: upper-left relative to the current view
    light.position
      .copy(lookTarget.current)
      .addScaledVector(toCamera.current, distance * 0.25)
      .addScaledVector(right.current, -distance * 0.55)
      .addScaledVector(camUp.current, distance * 0.85)

    light.target.position.copy(lookTarget.current)
    light.target.updateMatrixWorld()
  })

  return (
    <directionalLight ref={lightRef} intensity={intensity} color="#ffffff" castShadow>
      <object3D attach="target" />
    </directionalLight>
  )
}

function StudioLights() {
  return (
    <>
      <ambientLight intensity={1.8} color="#ffffff" />
      <KeyLightFollowCamera intensity={2.5} />
      <directionalLight position={[-100, 100, -100]} intensity={2.5} color="#ffffff" />
    </>
  )
}

function collectModelMeshes(root: Object3D): Mesh[] {
  const meshes: Mesh[] = []
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    if (child.name === 'selection-highlight') return
    if (child.name.startsWith('clip-stencil')) return
    if (child.name.startsWith('clip-cap')) return
    meshes.push(child as Mesh)
  })
  return meshes
}

function MeasureClickHandler({
  active,
  target,
  onHit,
}: {
  active: boolean
  target: Object3D | null
  onHit: (point: Vector3) => void
}) {
  const { camera, gl } = useThree()
  const raycaster = useMemo(() => new Raycaster(), [])
  const pointer = useMemo(() => new Vector2(), [])
  const onHitRef = useRef(onHit)
  onHitRef.current = onHit

  useEffect(() => {
    if (!active || !target) return

    const element = gl.domElement
    const onClick = (event: MouseEvent) => {
      const rect = element.getBoundingClientRect()
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      )
      raycaster.setFromCamera(pointer, camera)
      const hits = raycaster.intersectObjects(collectModelMeshes(target), false)
      if (hits[0]) onHitRef.current(hits[0].point.clone())
    }

    element.addEventListener('click', onClick)
    return () => element.removeEventListener('click', onClick)
  }, [active, target, camera, gl, pointer, raycaster])

  return null
}

function PartPickHandler({
  active,
  target,
  onPick,
}: {
  active: boolean
  target: Object3D | null
  onPick: (meshId: string) => void
}) {
  const { camera, gl } = useThree()
  const raycaster = useMemo(() => new Raycaster(), [])
  const pointer = useMemo(() => new Vector2(), [])
  const onPickRef = useRef(onPick)
  onPickRef.current = onPick
  const downPos = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    if (!active || !target) return

    const element = gl.domElement

    const onPointerDown = (event: PointerEvent) => {
      downPos.current = { x: event.clientX, y: event.clientY }
    }

    const onClick = (event: MouseEvent) => {
      if (downPos.current) {
        const dx = event.clientX - downPos.current.x
        const dy = event.clientY - downPos.current.y
        if (dx * dx + dy * dy > 36) return
      }

      const rect = element.getBoundingClientRect()
      pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      )
      raycaster.setFromCamera(pointer, camera)
      const hits = raycaster.intersectObjects(collectModelMeshes(target), false)
      const hit = hits.find((h) => h.object.visible && (h.object as Mesh).isMesh)
      if (hit) onPickRef.current(hit.object.uuid)
    }

    element.addEventListener('pointerdown', onPointerDown)
    element.addEventListener('click', onClick)
    return () => {
      element.removeEventListener('pointerdown', onPointerDown)
      element.removeEventListener('click', onClick)
    }
  }, [active, target, camera, gl, pointer, raycaster])

  return null
}

function SelectionHighlight({ mesh }: { mesh: Mesh | null }) {
  useEffect(() => {
    if (!mesh) return

    const edges = new EdgesGeometry(mesh.geometry, 25)
    const material = new LineBasicMaterial({
      color: '#4da3ff',
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    })
    const lines = new LineSegments(edges, material)
    lines.name = 'selection-highlight'
    lines.renderOrder = 10
    mesh.add(lines)

    return () => {
      mesh.remove(lines)
      edges.dispose()
      material.dispose()
    }
  }, [mesh])

  return null
}


function MeasurementMarkers({
  measurements,
  pending,
  markerScale,
}: {
  measurements: Measurement[]
  pending: [number, number, number] | null
  markerScale: number
}) {
  return (
    <group>
      {pending && (
        <mesh position={pending}>
          <sphereGeometry args={[markerScale, 18, 18]} />
          <meshBasicMaterial color="#ff2222" depthTest={false} />
        </mesh>
      )}

      {measurements.map((item) => {
        const a = new Vector3(...item.a)
        const b = new Vector3(...item.b)
        const mid = a.clone().add(b).multiplyScalar(0.5)
        const distance = a.distanceTo(b)

        return (
          <group key={item.id}>
            <mesh position={a}>
              <sphereGeometry args={[markerScale, 18, 18]} />
              <meshBasicMaterial color="#ff2222" depthTest={false} />
            </mesh>
            <mesh position={b}>
              <sphereGeometry args={[markerScale, 18, 18]} />
              <meshBasicMaterial color="#ff2222" depthTest={false} />
            </mesh>
            <Line points={[item.a, item.b]} color="#ff2222" lineWidth={2} />
            <Html position={mid} center distanceFactor={12} style={{ pointerEvents: 'none' }}>
              <div className="measure-label">{distance.toFixed(2)} mm</div>
            </Html>
          </group>
        )
      })}
    </group>
  )
}

function Scene({
  model,
  showHelpers,
  viewPreset,
  cameraRequestId,
  groundTexture,
  measureMode,
  measurements,
  pendingPoint,
  onMeasureHit,
  clipMasterEnabled,
  clipX,
  clipY,
  clipZ,
  clipDraggingAxis,
  showClipHelpers,
  selectedPartId,
  onPartPick,
  focusObject,
  focusRequestId,
  solidCapEnabled,
  capColor,
  capsRevision,
  orientationRef,
  viewDirRef,
  viewDirRequestId,
}: {
  model: LoadedModel | null
  showHelpers: boolean
  viewPreset: ViewPreset
  cameraRequestId: number
  groundTexture: Texture | null
  measureMode: boolean
  measurements: Measurement[]
  pendingPoint: [number, number, number] | null
  onMeasureHit: (point: Vector3) => void
  clipMasterEnabled: boolean
  clipX: ClipAxisState
  clipY: ClipAxisState
  clipZ: ClipAxisState
  clipDraggingAxis: Axis | null
  showClipHelpers: boolean
  selectedPartId: string | null
  onPartPick: (meshId: string) => void
  focusObject: Object3D | null
  focusRequestId: number
  solidCapEnabled: boolean
  capColor: string
  capsRevision: number
  orientationRef: MutableRefObject<Quaternion>
  viewDirRef: MutableRefObject<Vector3>
  viewDirRequestId: number
}) {
  const gridSize = useMemo(() => {
    if (!model) return 40
    const s = Math.max(model.stats.size.x, model.stats.size.z, 10)
    return Math.ceil(s * 2.5)
  }, [model])

  const markerScale = useMemo(() => {
    if (!model) return 0.4
    const diag = Math.hypot(model.stats.size.x, model.stats.size.y, model.stats.size.z)
    return Math.max(diag * 0.008, 0.15)
  }, [model])

  const selectedMesh = useMemo(() => {
    if (!model || !selectedPartId) return null
    return findMeshById(model.root, selectedPartId)
  }, [model, selectedPartId])

  return (
    <>
      <StudioLights />

      <GroundPlane size={gridSize} texture={groundTexture} />

      {showHelpers && (
        <>
          <gridHelper args={[gridSize, Math.min(gridSize, 40), '#4a4a4a', '#2c2c2c']} />
          <axesHelper args={[Math.max(gridSize * 0.15, 2)]} />
        </>
      )}

      {model && <primitive object={model.root} />}

      <ClippingController
        model={model}
        clipMasterEnabled={clipMasterEnabled}
        clipX={clipX}
        clipY={clipY}
        clipZ={clipZ}
        draggingAxis={clipDraggingAxis}
        showHelpersAlways={showClipHelpers}
        solidCapEnabled={solidCapEnabled}
        capColor={capColor}
        capsRevision={capsRevision}
      />

      <MeasureClickHandler
        active={measureMode && !!model}
        target={model?.root ?? null}
        onHit={onMeasureHit}
      />
      <PartPickHandler
        active={!measureMode && !!model}
        target={model?.root ?? null}
        onPick={onPartPick}
      />
      <SelectionHighlight mesh={selectedMesh} />
      <MeasurementMarkers
        measurements={measurements}
        pending={pendingPoint}
        markerScale={markerScale}
      />

      <CameraDirector
        target={model?.root ?? null}
        preset={viewPreset}
        requestId={cameraRequestId}
        focusObject={focusObject}
        focusRequestId={focusRequestId}
        viewDirRef={viewDirRef}
        viewDirRequestId={viewDirRequestId}
      />
      <OrbitControls makeDefault />
      <CameraOrientationPublisher orientationRef={orientationRef} />
    </>
  )
}

function formatLength(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (value >= 1000) return `${(value / 1000).toFixed(2)} m`
  if (value >= 10) return `${value.toFixed(1)} mm`
  if (value >= 1) return `${value.toFixed(2)} mm`
  return `${value.toFixed(3)} mm`
}

function formatTriangles(count: number): string {
  return count.toLocaleString('tr-TR')
}

function ToolbarButton({
  label,
  active,
  onClick,
  disabled,
  title,
}: {
  label: string
  active?: boolean
  onClick: () => void
  disabled?: boolean
  title?: string
}) {
  return (
    <button
      type="button"
      className={`tb-btn${active ? ' is-active' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={title ?? label}
    >
      {label}
    </button>
  )
}

export default function CADViewer() {
  const modelInputRef = useRef<HTMLInputElement>(null)
  const textureInputRef = useRef<HTMLInputElement>(null)
  const modelTextureUrlRef = useRef<string | null>(null)
  const groundTextureUrlRef = useRef<string | null>(null)
  const modelRef = useRef<LoadedModel | null>(null)
  const groundTextureRef = useRef<Texture | null>(null)

  const [model, setModel] = useState<LoadedModel | null>(null)
  const [textureTarget, setTextureTarget] = useState<TextureTarget>('model')
  const [modelTextureName, setModelTextureName] = useState<string | null>(null)
  const [groundTextureName, setGroundTextureName] = useState<string | null>(null)
  const [groundTexture, setGroundTexture] = useState<Texture | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('solid')
  const [showHelpers, setShowHelpers] = useState(true)
  const [viewPreset, setViewPreset] = useState<ViewPreset>('iso')
  const [cameraRequestId, setCameraRequestId] = useState(0)
  const [modelColor, setModelColor] = useState(DEFAULT_MODEL_COLOR)
  const [measureMode, setMeasureMode] = useState(false)
  const [measurements, setMeasurements] = useState<Measurement[]>([])
  const [pendingPoint, setPendingPoint] = useState<[number, number, number] | null>(null)
  const [clipPanelOpen, setClipPanelOpen] = useState(false)
  const [clipMasterEnabled, setClipMasterEnabled] = useState(false)
  const [clipX, setClipX] = useState<ClipAxisState>(DEFAULT_CLIP_AXIS)
  const [clipY, setClipY] = useState<ClipAxisState>(DEFAULT_CLIP_AXIS)
  const [clipZ, setClipZ] = useState<ClipAxisState>(DEFAULT_CLIP_AXIS)
  const [clipDraggingAxis, setClipDraggingAxis] = useState<Axis | null>(null)
  const [showClipHelpers, setShowClipHelpers] = useState(false)
  const [solidCapEnabled, setSolidCapEnabled] = useState(true)
  const [capColor, setCapColor] = useState('#8b0000')
  const [capsRevision, setCapsRevision] = useState(0)
  const [treePanelOpen, setTreePanelOpen] = useState(false)
  const [assemblyParts, setAssemblyParts] = useState<AssemblyPart[]>([])
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null)
  const [isolatedPartId, setIsolatedPartId] = useState<string | null>(null)
  const [focusObject, setFocusObject] = useState<Object3D | null>(null)
  const [focusRequestId, setFocusRequestId] = useState(0)
  const [viewDirRequestId, setViewDirRequestId] = useState(0)
  const orientationRef = useRef(new Quaternion())
  const viewDirRef = useRef(new Vector3(1, 1, 1).normalize())
  const dragDepth = useRef(0)
  const modelColorRef = useRef(modelColor)
  const pendingPointRef = useRef<[number, number, number] | null>(null)

  useEffect(() => {
    modelRef.current = model
  }, [model])

  useEffect(() => {
    modelColorRef.current = modelColor
  }, [modelColor])

  useEffect(() => {
    groundTextureRef.current = groundTexture
  }, [groundTexture])

  useEffect(() => {
    return () => {
      if (modelTextureUrlRef.current) URL.revokeObjectURL(modelTextureUrlRef.current)
      if (groundTextureUrlRef.current) URL.revokeObjectURL(groundTextureUrlRef.current)
      groundTextureRef.current?.dispose()
    }
  }, [])

  const revokeModelTextureUrl = useCallback(() => {
    if (modelTextureUrlRef.current) {
      URL.revokeObjectURL(modelTextureUrlRef.current)
      modelTextureUrlRef.current = null
    }
  }, [])

  const revokeGroundTextureUrl = useCallback(() => {
    if (groundTextureUrlRef.current) {
      URL.revokeObjectURL(groundTextureUrlRef.current)
      groundTextureUrlRef.current = null
    }
  }, [])

  const requestCameraView = useCallback((preset: ViewPreset) => {
    setViewPreset(preset)
    setCameraRequestId((id) => id + 1)
  }, [])

  const onViewCubeSelect = useCallback((direction: Vector3) => {
    viewDirRef.current.copy(direction).normalize()
    setViewDirRequestId((id) => id + 1)
  }, [])

  const applyModelTexture = useCallback(
    async (file: File) => {
      const current = modelRef.current
      if (!current) {
        setError('Model kaplaması için önce bir STL/OBJ modeli yükleyin.')
        return
      }

      setError(null)
      setStatus(`Model kaplaması yükleniyor: ${file.name}`)

      try {
        const { texture, url } = await loadTextureFromFile(file)
        revokeModelTextureUrl()
        modelTextureUrlRef.current = url
        applyTextureToModel(current.root, texture)
        applyViewMode(current.root, viewMode)
        setModelTextureName(file.name)
        setStatus(`Model kaplaması: ${file.name}`)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Kaplama yüklenemedi.'
        setError(message)
      }
    },
    [revokeModelTextureUrl, viewMode],
  )

  const applyGroundTexture = useCallback(
    async (file: File) => {
      setError(null)
      setStatus(`Zemin kaplaması yükleniyor: ${file.name}`)

      try {
        const { texture, url } = await loadTextureFromFile(file)
        prepareGroundTexture(texture)

        setGroundTexture((prev) => {
          prev?.dispose()
          return texture
        })
        revokeGroundTextureUrl()
        groundTextureUrlRef.current = url
        setGroundTextureName(file.name)
        setStatus(`Zemin kaplaması: ${file.name}`)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Kaplama yüklenemedi.'
        setError(message)
      }
    },
    [revokeGroundTextureUrl],
  )

  const handleTextureFile = useCallback(
    async (file: File) => {
      if (textureTarget === 'ground') {
        await applyGroundTexture(file)
        return
      }
      await applyModelTexture(file)
    },
    [applyGroundTexture, applyModelTexture, textureTarget],
  )

  const handleModelFile = useCallback(
    async (file: File) => {
      setError(null)
      setStatus(
        /\.(step|stp|iges|igs)$/i.test(file.name)
          ? `OpenCascade ile parse ediliyor: ${file.name}`
          : `Yükleniyor: ${file.name}`,
      )

      try {
        const loaded = await loadModelFromFile(file)
        applyModelColor(loaded.root, modelColorRef.current)
        applyViewMode(loaded.root, viewMode)
        revokeModelTextureUrl()
        setModelTextureName(null)
        setMeasurements([])
        setPendingPoint(null)
        pendingPointRef.current = null
        const bounds = getObjectClipBounds(loaded.root)
        setClipX({ enabled: false, value: bounds.max.x, inverted: false })
        setClipY({ enabled: false, value: bounds.max.y, inverted: false })
        setClipZ({ enabled: false, value: bounds.max.z, inverted: false })
        setClipMasterEnabled(false)
        const parts = buildAssemblyParts(loaded.root)
        setAssemblyParts(parts)
        setSelectedPartId(null)
        setIsolatedPartId(null)
        setFocusObject(null)
        setTreePanelOpen(parts.length > 0)
        setModel((prev) => {
          if (prev) disposeObject(prev.root)
          return loaded
        })
        setStatus(file.name)
        setViewPreset('iso')
        setCameraRequestId((id) => id + 1)
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Dosya yüklenemedi.'
        setError(message)
        setStatus(null)
      }
    },
    [revokeModelTextureUrl, viewMode],
  )

  const handleFile = useCallback(
    async (file: File | undefined | null) => {
      if (!file) return

      if (isTextureFile(file)) {
        await handleTextureFile(file)
        return
      }

      if (isModelFile(file)) {
        await handleModelFile(file)
        return
      }

      setError(
        'Desteklenen dosyalar: .stl, .obj, .step, .stp, .iges, .igs veya .jpg / .jpeg / .png / .webp',
      )
    },
    [handleModelFile, handleTextureFile],
  )

  const clearTexture = useCallback(() => {
    if (textureTarget === 'ground') {
      setGroundTexture((prev) => {
        prev?.dispose()
        return null
      })
      revokeGroundTextureUrl()
      setGroundTextureName(null)
      setStatus('Zemin kaplaması kaldırıldı')
      return
    }

    const current = modelRef.current
    if (!current) return
    clearTextureFromModel(current.root, modelColorRef.current)
    applyViewMode(current.root, viewMode)
    revokeModelTextureUrl()
    setModelTextureName(null)
    setStatus(current.name)
  }, [revokeGroundTextureUrl, revokeModelTextureUrl, textureTarget, viewMode])

  const onModelColorChange = (event: ChangeEvent<HTMLInputElement>) => {
    const color = event.target.value
    setModelColor(color)
    if (model) applyModelColor(model.root, color)
  }

  const onMeasureHit = useCallback((point: Vector3) => {
    const tuple: [number, number, number] = [point.x, point.y, point.z]
    const first = pendingPointRef.current

    if (!first) {
      pendingPointRef.current = tuple
      setPendingPoint(tuple)
      setStatus('İkinci noktayı seçin')
      return
    }

    pendingPointRef.current = null
    setPendingPoint(null)
    setMeasurements((list) => [
      ...list,
      {
        id: `m-${Date.now()}-${list.length}`,
        a: first,
        b: tuple,
      },
    ])
    setStatus('Ölçüm eklendi — yeni ölçüm için iki nokta daha seçin')
  }, [])

  const clearMeasurements = useCallback(() => {
    pendingPointRef.current = null
    setMeasurements([])
    setPendingPoint(null)
    setStatus('Ölçümler temizlendi')
  }, [])

  const toggleMeasureMode = useCallback(() => {
    setMeasureMode((active) => {
      const next = !active
      pendingPointRef.current = null
      setPendingPoint(null)
      if (next) {
        setStatus('Ölçüm modu: model üzerinde iki nokta seçin')
      } else {
        setStatus(null)
      }
      return next
    })
  }, [])

  const clipBounds = useMemo(() => getObjectClipBounds(model?.root ?? null), [model])

  const onClipAxisToggle = (axis: Axis, enabled: boolean) => {
    if (axis === 'x') setClipX((s) => ({ ...s, enabled }))
    if (axis === 'y') setClipY((s) => ({ ...s, enabled }))
    if (axis === 'z') setClipZ((s) => ({ ...s, enabled }))
  }

  const onClipValueChange = (axis: Axis, value: number) => {
    if (axis === 'x') setClipX((s) => ({ ...s, value }))
    if (axis === 'y') setClipY((s) => ({ ...s, value }))
    if (axis === 'z') setClipZ((s) => ({ ...s, value }))
  }

  const onClipInvert = (axis: Axis) => {
    if (axis === 'x') setClipX((s) => ({ ...s, inverted: !s.inverted }))
    if (axis === 'y') setClipY((s) => ({ ...s, inverted: !s.inverted }))
    if (axis === 'z') setClipZ((s) => ({ ...s, inverted: !s.inverted }))
  }

  const syncAssemblyVisibility = useCallback((root: Object3D) => {
    setAssemblyParts(buildAssemblyParts(root))
  }, [])

  const onPartPick = useCallback(
    (meshId: string) => {
      setSelectedPartId(meshId)
      setTreePanelOpen(true)
      const mesh = modelRef.current ? findMeshById(modelRef.current.root, meshId) : null
      if (mesh) {
        setFocusObject(mesh)
        setFocusRequestId((id) => id + 1)
        setStatus(`Seçili: ${mesh.name || meshId.slice(0, 8)}`)
      }
    },
    [],
  )

  const focusPart = useCallback((partId: string) => {
    const root = modelRef.current?.root
    if (!root) return
    const mesh = findMeshById(root, partId)
    if (!mesh) return
    setSelectedPartId(partId)
    setFocusObject(mesh)
    setFocusRequestId((id) => id + 1)
  }, [])

  const togglePartVisibility = useCallback((partId: string) => {
    const root = modelRef.current?.root
    if (!root) return
    const mesh = findMeshById(root, partId)
    if (!mesh) return
    mesh.visible = !mesh.visible
    if (isolatedPartId === partId && !mesh.visible) {
      setIsolatedPartId(null)
    }
    syncAssemblyVisibility(root)
    setCapsRevision((n) => n + 1)
  }, [isolatedPartId, syncAssemblyVisibility])

  const isolatePart = useCallback((partId: string) => {
    const root = modelRef.current?.root
    if (!root) return
    isolateMesh(root, partId)
    setIsolatedPartId(partId)
    setSelectedPartId(partId)
    syncAssemblyVisibility(root)
    setCapsRevision((n) => n + 1)
    const mesh = findMeshById(root, partId)
    if (mesh) {
      setFocusObject(mesh)
      setFocusRequestId((id) => id + 1)
    }
    setStatus('Parça izole edildi')
  }, [syncAssemblyVisibility])

  const showAllParts = useCallback(() => {
    const root = modelRef.current?.root
    if (!root) return
    showAllMeshes(root)
    setIsolatedPartId(null)
    syncAssemblyVisibility(root)
    setCapsRevision((n) => n + 1)
    setStatus('Tüm parçalar gösteriliyor')
  }, [syncAssemblyVisibility])

  const onViewModeChange = (mode: ViewMode) => {
    setViewMode(mode)
    if (model) applyViewMode(model.root, mode)
  }

  const onRotate = (axis: Axis) => {
    if (!model) return
    const next = rotateModelAxis(model, axis)
    applyViewMode(next.root, viewMode)
    setModel({ ...next })
    requestCameraView(viewPreset)
  }

  const onModelInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    void handleFile(file)
    event.target.value = ''
  }

  const onTextureInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) void handleTextureFile(file)
    event.target.value = ''
  }

  const onDragEnter = (event: DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    dragDepth.current += 1
    setIsDragging(true)
  }

  const onDragLeave = (event: DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (dragDepth.current === 0) setIsDragging(false)
  }

  const onDragOver = (event: DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
  }

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    event.stopPropagation()
    dragDepth.current = 0
    setIsDragging(false)
    void handleFile(event.dataTransfer.files?.[0])
  }

  const canUploadTexture = textureTarget === 'ground' || !!model
  const hasActiveTexture =
    textureTarget === 'ground' ? !!groundTextureName : !!modelTextureName

  return (
    <div
      className={`cad-viewer${measureMode ? ' is-measuring' : ''}`}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <Canvas
        className="main-viewport"
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
        camera={{ position: [4, 4, 4], fov: 50 }}
        gl={{ antialias: true, localClippingEnabled: true, stencil: true }}
        onCreated={({ gl }) => {
          gl.localClippingEnabled = true
        }}
      >
        <color attach="background" args={['#1a1a1a']} />
        <Scene
          model={model}
          showHelpers={showHelpers}
          viewPreset={viewPreset}
          cameraRequestId={cameraRequestId}
          groundTexture={groundTexture}
          measureMode={measureMode}
          measurements={measurements}
          pendingPoint={pendingPoint}
          onMeasureHit={onMeasureHit}
          clipMasterEnabled={clipMasterEnabled}
          clipX={clipX}
          clipY={clipY}
          clipZ={clipZ}
          clipDraggingAxis={clipDraggingAxis}
          showClipHelpers={showClipHelpers}
          selectedPartId={selectedPartId}
          onPartPick={onPartPick}
          focusObject={focusObject}
          focusRequestId={focusRequestId}
          solidCapEnabled={solidCapEnabled}
          capColor={capColor}
          capsRevision={capsRevision}
          orientationRef={orientationRef}
          viewDirRef={viewDirRef}
          viewDirRequestId={viewDirRequestId}
        />
      </Canvas>

      <ViewCube orientationRef={orientationRef} onSelectDirection={onViewCubeSelect} />

      <header className="cad-toolbar">
        <div className="tb-group">
          <span className="tb-label">Dosya</span>
          <ToolbarButton label="Dosya Yükle" onClick={() => modelInputRef.current?.click()} />
          <ToolbarButton
            label="Kaplama Seç"
            onClick={() => textureInputRef.current?.click()}
            disabled={!canUploadTexture}
            title={
              textureTarget === 'ground'
                ? 'Zemine JPG/PNG/WebP kaplama uygula'
                : 'Modele JPG/PNG/WebP kaplama uygula'
            }
          />
          {hasActiveTexture && (
            <ToolbarButton
              label="Kaplamayı Kaldır"
              onClick={clearTexture}
              title={
                textureTarget === 'ground'
                  ? 'Zemin kaplamasını temizle'
                  : 'Model kaplamasını temizle'
              }
            />
          )}
          <input
            ref={modelInputRef}
            type="file"
            accept=".stl,.obj,.step,.stp,.iges,.igs,model/stl,model/obj,application/sla,model/step,application/step"
            className="file-input"
            onChange={onModelInputChange}
          />
          <input
            ref={textureInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
            className="file-input"
            onChange={onTextureInputChange}
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Renk</span>
          <label className="color-picker" title="Model rengi">
            <input
              type="color"
              value={modelColor}
              onChange={onModelColorChange}
              disabled={!model}
              aria-label="Model rengi"
            />
            <span>{modelColor}</span>
          </label>
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Ölçüm</span>
          <ToolbarButton
            label="Ölçüm Yap"
            active={measureMode}
            onClick={toggleMeasureMode}
            disabled={!model}
            title="İki nokta arası mesafe ölç"
          />
          <ToolbarButton
            label="Ölçümleri Temizle"
            onClick={clearMeasurements}
            disabled={measurements.length === 0 && !pendingPoint}
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Kesit</span>
          <ToolbarButton
            label="Kesit Al"
            active={clipPanelOpen}
            onClick={() => setClipPanelOpen((v) => !v)}
            disabled={!model}
            title="3D kesit paneli"
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Montaj</span>
          <ToolbarButton
            label="Montaj Ağacı"
            active={treePanelOpen}
            onClick={() => setTreePanelOpen((v) => !v)}
            disabled={!model}
            title="Parça listesi / gizle-göster"
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Kapsam</span>
          <ToolbarButton
            label="Kapsam: Model"
            active={textureTarget === 'model'}
            onClick={() => setTextureTarget('model')}
            title="Kaplamayı modele uygula"
          />
          <ToolbarButton
            label="Kapsam: Zemin"
            active={textureTarget === 'ground'}
            onClick={() => setTextureTarget('ground')}
            title="Kaplamayı zemine uygula"
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Görünüm</span>
          <ToolbarButton
            label="Katı"
            active={viewMode === 'solid'}
            onClick={() => onViewModeChange('solid')}
          />
          <ToolbarButton
            label="Tel Kafes"
            active={viewMode === 'wireframe'}
            onClick={() => onViewModeChange('wireframe')}
          />
          <ToolbarButton
            label="Şeffaf"
            active={viewMode === 'translucent'}
            onClick={() => onViewModeChange('translucent')}
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Kamera</span>
          <ToolbarButton
            label="İzo"
            active={viewPreset === 'iso'}
            onClick={() => requestCameraView('iso')}
            disabled={!model}
            title="İzometrik"
          />
          <ToolbarButton
            label="Üst"
            active={viewPreset === 'top'}
            onClick={() => requestCameraView('top')}
            disabled={!model}
          />
          <ToolbarButton
            label="Ön"
            active={viewPreset === 'front'}
            onClick={() => requestCameraView('front')}
            disabled={!model}
          />
          <ToolbarButton
            label="Sağ"
            active={viewPreset === 'right'}
            onClick={() => requestCameraView('right')}
            disabled={!model}
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Döndür</span>
          <ToolbarButton
            label="X 90°"
            onClick={() => onRotate('x')}
            disabled={!model}
            title="X ekseninde 90° döndür"
          />
          <ToolbarButton
            label="Y 90°"
            onClick={() => onRotate('y')}
            disabled={!model}
            title="Y ekseninde 90° döndür"
          />
          <ToolbarButton
            label="Z 90°"
            onClick={() => onRotate('z')}
            disabled={!model}
            title="Z ekseninde 90° döndür"
          />
        </div>

        <div className="tb-divider" />

        <div className="tb-group">
          <span className="tb-label">Sahne</span>
          <ToolbarButton
            label={showHelpers ? 'Grid Açık' : 'Grid Kapalı'}
            active={showHelpers}
            onClick={() => setShowHelpers((v) => !v)}
            title="Grid ve eksenleri aç/kapat"
          />
        </div>

        {(status || error) && (
          <div className="tb-status">
            {error ? <span className="is-error">{error}</span> : <span>{status}</span>}
          </div>
        )}
      </header>

      {treePanelOpen && model && (
        <aside className="assembly-panel">
          <div className="assembly-panel-header">
            <div>
              <strong>Montaj Ağacı</strong>
              <p>{assemblyParts.length} parça</p>
            </div>
            <button
              type="button"
              className="tb-btn"
              onClick={() => setTreePanelOpen(false)}
              title="Paneli kapat"
            >
              Kapat
            </button>
          </div>

          {isolatedPartId && (
            <button type="button" className="tb-btn assembly-show-all" onClick={showAllParts}>
              Tümünü Göster
            </button>
          )}

          <ul className="assembly-list">
            {assemblyParts.map((part) => (
              <li
                key={part.id}
                className={`assembly-item${selectedPartId === part.id ? ' is-selected' : ''}${
                  !part.visible ? ' is-hidden' : ''
                }`}
              >
                <button
                  type="button"
                  className={`assembly-eye${part.visible ? '' : ' is-off'}`}
                  title={part.visible ? 'Gizle' : 'Göster'}
                  onClick={() => togglePartVisibility(part.id)}
                  aria-label={part.visible ? 'Gizle' : 'Göster'}
                >
                  <span className="eye-glyph" />
                </button>
                <button
                  type="button"
                  className="assembly-name"
                  title="Parçaya odaklan"
                  onClick={() => focusPart(part.id)}
                >
                  {part.name}
                </button>
                <button
                  type="button"
                  className={`tb-btn assembly-isolate${
                    isolatedPartId === part.id ? ' is-active' : ''
                  }`}
                  title="İzole et"
                  onClick={() => isolatePart(part.id)}
                >
                  İzole
                </button>
              </li>
            ))}
          </ul>
        </aside>
      )}

      {clipPanelOpen && model && (
        <aside className={`clip-panel${treePanelOpen ? ' with-tree' : ''}`}>
          <div className="clip-panel-header">
            <strong>Kesit Kontrolü</strong>
            <button
              type="button"
              className="tb-btn"
              onClick={() => setClipPanelOpen(false)}
              title="Paneli kapat"
            >
              Kapat
            </button>
          </div>

          <label className="clip-row clip-master">
            <input
              type="checkbox"
              checked={clipMasterEnabled}
              onChange={(e) => setClipMasterEnabled(e.target.checked)}
            />
            <span>Kesit Modu Aç/Kapat</span>
          </label>

          <label className="clip-row">
            <input
              type="checkbox"
              checked={showClipHelpers}
              onChange={(e) => setShowClipHelpers(e.target.checked)}
            />
            <span>Yardımcı düzlemleri göster</span>
          </label>

          <label className="clip-row clip-master">
            <input
              type="checkbox"
              checked={solidCapEnabled}
              onChange={(e) => setSolidCapEnabled(e.target.checked)}
              disabled={!clipMasterEnabled}
            />
            <span>Kesit Yüzeyini Doldur (Solid Cap)</span>
          </label>

          <label className={`color-picker clip-cap-color${!solidCapEnabled || !clipMasterEnabled ? ' is-disabled' : ''}`}>
            <input
              type="color"
              value={capColor}
              disabled={!solidCapEnabled || !clipMasterEnabled}
              onChange={(e) => setCapColor(e.target.value)}
              aria-label="Kesit kapak rengi"
            />
            <span>Kapak rengi</span>
          </label>

          {([
            { axis: 'x' as const, label: 'X Ekseni', state: clipX, min: clipBounds.min.x, max: clipBounds.max.x, color: '#ff6b6b' },
            { axis: 'y' as const, label: 'Y Ekseni', state: clipY, min: clipBounds.min.y, max: clipBounds.max.y, color: '#6bff9a' },
            { axis: 'z' as const, label: 'Z Ekseni', state: clipZ, min: clipBounds.min.z, max: clipBounds.max.z, color: '#6ba8ff' },
          ]).map(({ axis, label, state, min, max, color }) => (
            <div key={axis} className={`clip-axis${state.enabled && clipMasterEnabled ? ' is-on' : ''}`}>
              <div className="clip-axis-top">
                <label className="clip-row">
                  <input
                    type="checkbox"
                    checked={state.enabled}
                    disabled={!clipMasterEnabled}
                    onChange={(e) => onClipAxisToggle(axis, e.target.checked)}
                  />
                  <span style={{ color }}>{label}</span>
                </label>
                <button
                  type="button"
                  className={`tb-btn${state.inverted ? ' is-active' : ''}`}
                  disabled={!clipMasterEnabled || !state.enabled}
                  onClick={() => onClipInvert(axis)}
                  title="Kesit yönünü ters çevir"
                >
                  Yönü Ters Çevir
                </button>
              </div>
              <input
                type="range"
                className="clip-slider"
                min={min}
                max={max}
                step={(max - min) / 500 || 0.01}
                value={Math.min(max, Math.max(min, state.value))}
                disabled={!clipMasterEnabled || !state.enabled}
                onPointerDown={() => setClipDraggingAxis(axis)}
                onPointerUp={() => setClipDraggingAxis(null)}
                onPointerCancel={() => setClipDraggingAxis(null)}
                onChange={(e) => onClipValueChange(axis, Number(e.target.value))}
              />
              <div className="clip-axis-value">{state.value.toFixed(2)}</div>
            </div>
          ))}
        </aside>
      )}

      {(model || groundTextureName) && (
        <aside className="model-info">
          {model ? (
            <>
              <div className="model-info-title">{model.name}</div>
              <div className="model-info-row">
                <span>Üçgen</span>
                <strong>{formatTriangles(model.stats.triangles)}</strong>
              </div>
              <div className="model-info-row">
                <span>X</span>
                <strong>{formatLength(model.stats.size.x)}</strong>
              </div>
              <div className="model-info-row">
                <span>Y</span>
                <strong>{formatLength(model.stats.size.y)}</strong>
              </div>
              <div className="model-info-row">
                <span>Z</span>
                <strong>{formatLength(model.stats.size.z)}</strong>
              </div>
            </>
          ) : (
            <div className="model-info-title">Sahne</div>
          )}
          {modelTextureName && (
            <div className="model-info-row">
              <span>Model</span>
              <strong className="texture-name">{modelTextureName}</strong>
            </div>
          )}
          {groundTextureName && (
            <div className="model-info-row">
              <span>Zemin</span>
              <strong className="texture-name">{groundTextureName}</strong>
            </div>
          )}
          <p className="model-info-note">Boyutlar birim olarak (genelde mm)</p>
        </aside>
      )}

      {isDragging && (
        <div className="drop-overlay" aria-hidden>
          <div className="drop-panel">
            <span className="drop-title">Dosyayı bırakın</span>
            <span className="drop-hint">
              {textureTarget === 'ground'
                ? 'Kaplama → Zemin (.jpg / .png)'
                : 'STL / OBJ / STEP / IGES veya Model kaplaması'}
            </span>
          </div>
        </div>
      )}

      {!isDragging && (
        <p className="empty-hint">
          STL / OBJ / STEP / IGES modellerinizi veya kaplamak istediğiniz Görsel (JPG/PNG)
          dosyalarını buraya sürükleyin. Kaplama hedefi:{' '}
          {textureTarget === 'ground' ? 'Zemin' : 'Model'}.
        </p>
      )}
    </div>
  )
}
