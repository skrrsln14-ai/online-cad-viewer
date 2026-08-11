import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  EdgesGeometry,
  Group,
  LineBasicMaterial,
  LineSegments,
  Material,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  RepeatWrapping,
  Sphere,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  Vector3,
  DirectionalLight,
} from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { loadOcctContent } from '../lib/loadOcctModel'
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
  color: '#d0d7de',
  metalness: 0.2,
  roughness: 0.5,
} as const

const EDGE_THRESHOLD_DEG = 35
const EDGE_COLOR = '#1a1a1a'

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

function clearTextureFromModel(root: Object3D) {
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
      mat.color.set(BASE_MATERIAL.color)
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
}: {
  target: Object3D | null
  preset: ViewPreset
  requestId: number
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

  useEffect(() => {
    if (!target || requestId === 0) return

    const box = new Box3().setFromObject(target)
    if (box.isEmpty()) return

    const center = box.getCenter(new Vector3())
    const radius = Math.max(box.getBoundingSphere(new Sphere()).radius, 0.01)
    const fov = 'fov' in camera ? ((camera.fov as number) * Math.PI) / 180 : Math.PI / 4
    const distance = (radius / Math.sin(fov / 2)) * 1.2
    const toPos = center.clone().add(VIEW_DIRECTIONS[preset].clone().multiplyScalar(distance))

    camera.near = Math.max(distance / 100, 0.01)
    camera.far = Math.max(distance * 50, 100)
    camera.updateProjectionMatrix()

    anim.current = {
      t: 0,
      duration: 0.55,
      fromPos: camera.position.clone(),
      toPos,
      fromTarget: controls?.target.clone() ?? center.clone(),
      toTarget: center.clone(),
    }

    if (controls) controls.enabled = false
  }, [target, preset, requestId, camera, controls])

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
      <ambientLight intensity={1.75} color="#ffffff" />
      <hemisphereLight args={['#ffffff', '#444444', 1.0]} />
      <KeyLightFollowCamera intensity={2.5} />
      {/* Fill — softens shadows from camera-right / back */}
      <directionalLight position={[-100, 100, -100]} intensity={1.5} color="#ffffff" />
    </>
  )
}

function Scene({
  model,
  showHelpers,
  viewPreset,
  cameraRequestId,
  groundTexture,
}: {
  model: LoadedModel | null
  showHelpers: boolean
  viewPreset: ViewPreset
  cameraRequestId: number
  groundTexture: Texture | null
}) {
  const gridSize = useMemo(() => {
    if (!model) return 40
    const s = Math.max(model.stats.size.x, model.stats.size.z, 10)
    return Math.ceil(s * 2.5)
  }, [model])

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

      <CameraDirector
        target={model?.root ?? null}
        preset={viewPreset}
        requestId={cameraRequestId}
      />
      <OrbitControls makeDefault />
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
  const dragDepth = useRef(0)

  useEffect(() => {
    modelRef.current = model
  }, [model])

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
        applyViewMode(loaded.root, viewMode)
        revokeModelTextureUrl()
        setModelTextureName(null)
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
    clearTextureFromModel(current.root)
    applyViewMode(current.root, viewMode)
    revokeModelTextureUrl()
    setModelTextureName(null)
    setStatus(current.name)
  }, [revokeGroundTextureUrl, revokeModelTextureUrl, textureTarget, viewMode])

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
      className="cad-viewer"
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <Canvas
        style={{ width: '100%', height: '100%' }}
        camera={{ position: [4, 4, 4], fov: 50 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={['#1a1a1a']} />
        <Scene
          model={model}
          showHelpers={showHelpers}
          viewPreset={viewPreset}
          cameraRequestId={cameraRequestId}
          groundTexture={groundTexture}
        />
      </Canvas>

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
