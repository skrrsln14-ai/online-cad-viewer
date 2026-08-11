import {
  BufferAttribute,
  BufferGeometry,
  CylinderGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Raycaster,
  Vector3,
} from 'three'

export type SupportGeneratorOptions = {
  /** Overhang angle from horizontal (30–60). Needs support when steeper downward. */
  overhangAngleDeg: number
  /** 0 = sparse, 1 = dense */
  density: number
  /** Pillar radius in model units (mm). */
  pillarRadius: number
}

const SUPPORT_GROUP_NAME = 'print-supports'
const SUPPORT_PILLAR_NAME = 'support-pillar'

const SUPPORT_MATERIAL = () =>
  new MeshStandardMaterial({
    color: '#00bcd4',
    transparent: true,
    opacity: 0.8,
    metalness: 0.05,
    roughness: 0.65,
    depthWrite: false,
  })

function collectSupportMeshes(root: Object3D): Mesh[] {
  const meshes: Mesh[] = []
  root.traverse((child) => {
    if (!(child as Mesh).isMesh) return
    if (child.name === 'selection-highlight') return
    if (child.name === SUPPORT_PILLAR_NAME) return
    if (child.name.startsWith('clip-stencil')) return
    if (child.name.startsWith('clip-cap')) return
    if (child.name === 'cad-edges') return
    meshes.push(child as Mesh)
  })
  return meshes
}

/** normal·up < -sin(θ) ⇔ angle from up > 90°+θ (downward overhang). */
function overhangDotThreshold(angleDeg: number): number {
  return -Math.sin((angleDeg * Math.PI) / 180)
}

function triangleArea(a: Vector3, b: Vector3, c: Vector3): number {
  const ab = new Vector3().subVectors(b, a)
  const ac = new Vector3().subVectors(c, a)
  return ab.cross(ac).length() * 0.5
}

/**
 * Sample overhang face centroids in world space.
 * Density controls min spacing between support points.
 */
export function findOverhangSupportPoints(
  root: Object3D,
  overhangAngleDeg: number,
  density: number,
): Vector3[] {
  root.updateMatrixWorld(true)
  const threshold = overhangDotThreshold(overhangAngleDeg)
  const worldNormal = new Vector3()
  const a = new Vector3()
  const b = new Vector3()
  const c = new Vector3()
  const centroid = new Vector3()

  type Candidate = { point: Vector3; area: number }
  const candidates: Candidate[] = []

  const meshes = collectSupportMeshes(root)
  for (const mesh of meshes) {
    if (!mesh.visible) continue
    const geometry = mesh.geometry as BufferGeometry
    const pos = geometry.getAttribute('position') as BufferAttribute | undefined
    if (!pos) continue

    const index = geometry.getIndex()
    const matrixWorld = mesh.matrixWorld

    const triCount = index ? index.count / 3 : pos.count / 3
    for (let t = 0; t < triCount; t++) {
      let i0: number
      let i1: number
      let i2: number
      if (index) {
        i0 = index.getX(t * 3)
        i1 = index.getX(t * 3 + 1)
        i2 = index.getX(t * 3 + 2)
      } else {
        i0 = t * 3
        i1 = t * 3 + 1
        i2 = t * 3 + 2
      }

      a.fromBufferAttribute(pos, i0).applyMatrix4(matrixWorld)
      b.fromBufferAttribute(pos, i1).applyMatrix4(matrixWorld)
      c.fromBufferAttribute(pos, i2).applyMatrix4(matrixWorld)

      // Face normal from world positions (more reliable than transformed normals)
      worldNormal.subVectors(b, a).cross(c.clone().sub(a)).normalize()
      if (worldNormal.y >= threshold) continue

      const area = triangleArea(a, b, c)
      if (area < 1e-8) continue

      centroid
        .copy(a)
        .add(b)
        .add(c)
        .multiplyScalar(1 / 3)

      // Skip near-ground faces (already on build plate)
      if (centroid.y < Math.max(0.15, Math.abs(worldNormal.y) * 0.5)) continue

      candidates.push({ point: centroid.clone(), area })
    }
  }

  if (candidates.length === 0) return []

  // Larger faces first — denser packing with higher density
  candidates.sort((x, y) => y.area - x.area)

  let bboxDiag = 10
  const boxMin = new Vector3(Infinity, Infinity, Infinity)
  const boxMax = new Vector3(-Infinity, -Infinity, -Infinity)
  for (const { point } of candidates) {
    boxMin.min(point)
    boxMax.max(point)
  }
  bboxDiag = Math.max(boxMax.distanceTo(boxMin), 1)

  const dens = Math.min(1, Math.max(0, density))
  // Sparse → ~12% of diagonal spacing; dense → ~2%
  const minSpacing = bboxDiag * (0.12 - dens * 0.1)

  const selected: Vector3[] = []
  for (const { point } of candidates) {
    let ok = true
    for (const existing of selected) {
      if (existing.distanceTo(point) < minSpacing) {
        ok = false
        break
      }
    }
    if (ok) selected.push(point)
  }

  return selected
}

function disposeSupportGroup(group: Object3D) {
  group.traverse((child) => {
    const mesh = child as Mesh
    if (!mesh.isMesh) return
    mesh.geometry?.dispose()
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    mats.forEach((m) => m?.dispose())
  })
}

export function clearSupportPillars(parent: Object3D | null | undefined) {
  if (!parent) return
  const existing = parent.getObjectByName(SUPPORT_GROUP_NAME)
  if (!existing) return
  parent.remove(existing)
  disposeSupportGroup(existing)
}

/**
 * Build cyan translucent pillars from overhang points down to ground or a surface below.
 */
export function generateSupportPillars(
  modelRoot: Object3D,
  options: SupportGeneratorOptions,
  attachTo?: Object3D,
): { group: Group; count: number } {
  const parent = attachTo ?? modelRoot.parent ?? modelRoot
  clearSupportPillars(parent)

  const points = findOverhangSupportPoints(
    modelRoot,
    options.overhangAngleDeg,
    options.density,
  )

  const group = new Group()
  group.name = SUPPORT_GROUP_NAME

  const meshes = collectSupportMeshes(modelRoot)
  const raycaster = new Raycaster()
  const down = new Vector3(0, -1, 0)
  const origin = new Vector3()
  const material = SUPPORT_MATERIAL()
  const radius = Math.max(options.pillarRadius, 0.05)

  let count = 0
  for (const point of points) {
    origin.copy(point)
    origin.y -= Math.max(radius * 0.25, 0.02)
    raycaster.set(origin, down)
    raycaster.far = point.y + 1

    const hits = raycaster.intersectObjects(meshes, false)
    let bottomY = 0
    for (const hit of hits) {
      if (hit.distance < Math.max(radius * 0.5, 0.05)) continue
      bottomY = Math.max(0, hit.point.y)
      break
    }

    const topY = point.y
    const height = topY - bottomY
    if (height < radius * 0.75) continue

    const cylinder = new Mesh(
      new CylinderGeometry(radius, radius * 0.85, height, 10),
      material.clone(),
    )
    cylinder.name = SUPPORT_PILLAR_NAME
    cylinder.position.set(point.x, bottomY + height / 2, point.z)
    cylinder.castShadow = false
    cylinder.receiveShadow = false
    cylinder.raycast = () => undefined
    group.add(cylinder)
    count += 1
  }

  // Shared material on first pillar is cloned; dispose unused base
  material.dispose()

  parent.add(group)
  return { group, count }
}

export function isSupportObject(obj: Object3D): boolean {
  return obj.name === SUPPORT_GROUP_NAME || obj.name === SUPPORT_PILLAR_NAME
}
