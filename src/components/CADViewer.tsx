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
  BufferGeometry,
  Group,
  Material,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Sphere,
  Vector3,
} from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import './CADViewer.css'

type ViewMode = 'solid' | 'wireframe' | 'translucent'
type ViewPreset = 'iso' | 'top' | 'front' | 'right'
type Axis = 'x' | 'y' | 'z'

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
  color: '#9a9a9a',
  metalness: 0.75,
  roughness: 0.35,
} as const

const ACCEPTED_EXTENSIONS = ['.stl', '.obj'] as const

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

function isAcceptedFile(file: File): boolean {
  return (ACCEPTED_EXTENSIONS as readonly string[]).includes(getExtension(file.name))
}

function createMaterial(): MeshStandardMaterial {
  return new MeshStandardMaterial({ ...BASE_MATERIAL })
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
  } else {
    throw new Error(`Desteklenmeyen dosya türü: ${ext || 'bilinmiyor'}`)
  }

  const root = wrapLoadedObject(content)
  return { root, name: file.name, stats: computeStats(root) }
}

function disposeObject(root: Object3D) {
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    const mesh = child as Mesh
    mesh.geometry?.dispose()
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    materials.forEach((mat: Material) => mat.dispose())
  })
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
      mat.metalness = mode === 'wireframe' ? 0.2 : BASE_MATERIAL.metalness
      mat.roughness = mode === 'wireframe' ? 0.6 : BASE_MATERIAL.roughness
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

function Scene({
  model,
  showHelpers,
  viewPreset,
  cameraRequestId,
}: {
  model: LoadedModel | null
  showHelpers: boolean
  viewPreset: ViewPreset
  cameraRequestId: number
}) {
  const gridSize = useMemo(() => {
    if (!model) return 40
    const s = Math.max(model.stats.size.x, model.stats.size.z, 10)
    return Math.ceil(s * 2.5)
  }, [model])

  return (
    <>
      <ambientLight intensity={0.45} />
      <directionalLight position={[6, 10, 4]} intensity={1.15} />
      <directionalLight position={[-4, 3, -6]} intensity={0.35} />

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
  const inputRef = useRef<HTMLInputElement>(null)
  const [model, setModel] = useState<LoadedModel | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('solid')
  const [showHelpers, setShowHelpers] = useState(true)
  const [viewPreset, setViewPreset] = useState<ViewPreset>('iso')
  const [cameraRequestId, setCameraRequestId] = useState(0)
  const dragDepth = useRef(0)

  const requestCameraView = useCallback((preset: ViewPreset) => {
    setViewPreset(preset)
    setCameraRequestId((id) => id + 1)
  }, [])

  const handleFile = useCallback(
    async (file: File | undefined | null) => {
      if (!file) return

      if (!isAcceptedFile(file)) {
        setError('Yalnızca .stl veya .obj dosyaları desteklenir.')
        return
      }

      setError(null)
      setStatus(`Yükleniyor: ${file.name}`)

      try {
        const loaded = await loadModelFromFile(file)
        applyViewMode(loaded.root, viewMode)
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
    [viewMode],
  )

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

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    void handleFile(file)
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
        />
      </Canvas>

      <header className="cad-toolbar">
        <div className="tb-group">
          <span className="tb-label">Dosya</span>
          <ToolbarButton label="Dosya Yükle" onClick={() => inputRef.current?.click()} />
          <input
            ref={inputRef}
            type="file"
            accept=".stl,.obj,model/stl,model/obj,application/sla"
            className="file-input"
            onChange={onInputChange}
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

      {model && (
        <aside className="model-info">
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
          <p className="model-info-note">Boyutlar birim olarak (genelde mm)</p>
        </aside>
      )}

      {isDragging && (
        <div className="drop-overlay" aria-hidden>
          <div className="drop-panel">
            <span className="drop-title">Dosyayı bırakın</span>
            <span className="drop-hint">.stl veya .obj</span>
          </div>
        </div>
      )}

      {!model && !isDragging && (
        <p className="empty-hint">
          STL / OBJ dosyasını sürükleyip bırakın veya Dosya Yükle’ye tıklayın
        </p>
      )}
    </div>
  )
}
