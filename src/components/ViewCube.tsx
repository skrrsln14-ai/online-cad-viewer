import { useMemo, useRef, useState, type MutableRefObject, type RefObject } from 'react'
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber'
import {
  CanvasTexture,
  Group,
  Quaternion,
  Vector3,
} from 'three'

const FACES = ['SAĞ', 'SOL', 'ÜST', 'ALT', 'ÖN', 'ARKA'] as const

const STYLE = {
  color: '#2a3038',
  hoverColor: '#6eb6ff',
  textColor: '#f0f3f7',
  strokeColor: '#0c0e12',
  opacity: 0.92,
  font: 'bold 20px "Segoe UI", system-ui, sans-serif',
} as const

const makePos = (xyz: [number, number, number]) => new Vector3(...xyz).multiplyScalar(0.38)

const CORNERS = (
  [
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
  ] as [number, number, number][]
).map(makePos)

const EDGES = (
  [
    [1, 1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [1, -1, 0],
    [0, 1, 1],
    [0, 1, -1],
    [0, -1, 1],
    [0, -1, -1],
    [-1, 1, 0],
    [-1, 0, 1],
    [-1, 0, -1],
    [-1, -1, 0],
  ] as [number, number, number][]
).map(makePos)

const EDGE_DIMS = EDGES.map((edge) => edge.toArray().map((axis) => (axis === 0 ? 0.5 : 0.25)) as [number, number, number])
const CORNER_DIMS: [number, number, number] = [0.25, 0.25, 0.25]

function makeFaceTexture(index: number): CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 128
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = STYLE.color
  ctx.fillRect(0, 0, 128, 128)
  ctx.strokeStyle = STYLE.strokeColor
  ctx.lineWidth = 3
  ctx.strokeRect(2, 2, 124, 124)
  ctx.font = STYLE.font
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = STYLE.textColor
  ctx.fillText(FACES[index], 64, 64)
  const texture = new CanvasTexture(canvas)
  texture.needsUpdate = true
  return texture
}

function FaceMaterial({
  index,
  hover,
}: {
  index: number
  hover: boolean
}) {
  const gl = useThree((s) => s.gl)
  const texture = useMemo(() => makeFaceTexture(index), [index])

  return (
    <meshBasicMaterial
      attach={`material-${index}`}
      map={texture}
      map-anisotropy={gl.capabilities.getMaxAnisotropy() || 1}
      color={hover ? STYLE.hoverColor : '#ffffff'}
      transparent
      opacity={STYLE.opacity}
      toneMapped={false}
    />
  )
}

function HitBox({
  position,
  dimensions,
  onSelect,
}: {
  position: Vector3
  dimensions: [number, number, number]
  onSelect: (direction: Vector3) => void
}) {
  const [hover, setHover] = useState(false)

  return (
    <mesh
      scale={1.01}
      position={position}
      onPointerOver={(e) => {
        e.stopPropagation()
        setHover(true)
      }}
      onPointerOut={(e) => {
        e.stopPropagation()
        setHover(false)
      }}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(position.clone().normalize())
      }}
    >
      <boxGeometry args={dimensions} />
      <meshBasicMaterial
        color={hover ? STYLE.hoverColor : '#ffffff'}
        transparent
        opacity={hover ? 0.55 : 0}
        depthWrite={false}
        toneMapped={false}
        visible={hover}
      />
    </mesh>
  )
}

function ViewCubeMesh({ onSelect }: { onSelect: (direction: Vector3) => void }) {
  const [hoverFace, setHoverFace] = useState<number | null>(null)

  const onFaceMove = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    if (e.faceIndex == null) return
    setHoverFace(Math.floor(e.faceIndex / 2))
  }

  const onFaceOut = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation()
    setHoverFace(null)
  }

  const onFaceClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    if (!e.face) return
    onSelect(e.face.normal.clone())
  }

  return (
    <group scale={1.15}>
      <mesh onPointerMove={onFaceMove} onPointerOut={onFaceOut} onClick={onFaceClick}>
        <boxGeometry />
        {FACES.map((_, index) => (
          <FaceMaterial key={index} index={index} hover={hoverFace === index} />
        ))}
      </mesh>

      {EDGES.map((edge, i) => (
        <HitBox key={`e-${i}`} position={edge} dimensions={EDGE_DIMS[i]} onSelect={onSelect} />
      ))}
      {CORNERS.map((corner, i) => (
        <HitBox key={`c-${i}`} position={corner} dimensions={CORNER_DIMS} onSelect={onSelect} />
      ))}
    </group>
  )
}

function ViewCubeScene({
  orientationRef,
  onSelectDirection,
}: {
  orientationRef: RefObject<Quaternion>
  onSelectDirection: (direction: Vector3) => void
}) {
  const groupRef = useRef<Group>(null)

  useFrame(() => {
    const group = groupRef.current
    const q = orientationRef.current
    if (!group || !q) return
    group.quaternion.copy(q)
  })

  return (
    <>
      <ambientLight intensity={1.2} />
      <group ref={groupRef}>
        <ViewCubeMesh onSelect={onSelectDirection} />
      </group>
    </>
  )
}

export type ViewCubeProps = {
  orientationRef: MutableRefObject<Quaternion>
  onSelectDirection: (direction: Vector3) => void
}

export default function ViewCube({ orientationRef, onSelectDirection }: ViewCubeProps) {
  return (
    <div className="viewcube-root">
      <Canvas
        className="viewcube-canvas"
        orthographic
        camera={{ position: [0, 0, 4], zoom: 48, near: 0.1, far: 100 }}
        frameloop="always"
        dpr={[1, 1.75]}
        gl={{
          alpha: true,
          antialias: true,
          premultipliedAlpha: false,
        }}
        onCreated={({ gl }) => {
          gl.setClearColor(0x000000, 0)
          gl.setClearAlpha(0)
        }}
        style={{ width: '100%', height: '100%', background: 'transparent', pointerEvents: 'auto' }}
      >
        <ViewCubeScene orientationRef={orientationRef} onSelectDirection={onSelectDirection} />
      </Canvas>
    </div>
  )
}

/** Publishes inverted main-camera world rotation for ViewCube sync. */
export function CameraOrientationPublisher({
  orientationRef,
}: {
  orientationRef: MutableRefObject<Quaternion>
}) {
  const camera = useThree((s) => s.camera)
  const scratch = useMemo(() => ({ mat: camera.matrixWorld.clone() }), [camera])

  useFrame(() => {
    scratch.mat.copy(camera.matrixWorld).invert()
    orientationRef.current.setFromRotationMatrix(scratch.mat)
  })

  return null
}
