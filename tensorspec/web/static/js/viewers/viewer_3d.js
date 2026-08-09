/* Crystal viewport: renders the geometry payload from /api/crystal.
 *
 * This module draws only. Every position, radius and bond pair is decided by
 * `core/crystallography.py`; nothing here computes geometry. Colours are the
 * one exception, being a display choice rather than a property of the crystal.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { pbrMaterialParams } from "./pbr_params.js";

const CPK_COLORS = {
  H:"#FFFFFF", He:"#D9FFFF", Li:"#CC80FF", Be:"#C2FF00", B:"#FFB5B5",
  C:"#909090", N:"#3050F8", O:"#FF0D0D", F:"#90E050", Ne:"#B3E3F5",
  Na:"#AB5CF2", Mg:"#8AFF00", Al:"#BFA6A6", Si:"#F0C8A0", P:"#FF8000",
  S:"#FFFF30", Cl:"#1FF01F", Ar:"#80D1E3", K:"#8F40D4", Ca:"#3DFF00",
  Sc:"#E6E6E6", Ti:"#BFC2C7", V:"#A6A6AB", Cr:"#8A99C7", Mn:"#9C7AC7",
  Fe:"#E06633", Co:"#F090A0", Ni:"#50D050", Cu:"#C88033", Zn:"#7D80B0",
  Ga:"#C28F8F", Ge:"#668F8F", As:"#BD80E3", Se:"#FFA100", Br:"#A62929",
  Kr:"#5CB8D1", Rb:"#702EB0", Sr:"#00FF00", Y:"#94FFFF", Zr:"#94E0E0",
  Nb:"#73C2C9", Mo:"#54B5B5", Tc:"#3B9E9E", Ru:"#248F8F", Rh:"#0A7D8C",
  Pd:"#006985", Ag:"#C0C0C0", Cd:"#FFD98F", In:"#A67573", Sn:"#668080",
  Sb:"#9E63B5", Te:"#D47A00", I:"#940094", Xe:"#429EB0", Cs:"#57178F",
  Ba:"#00C900", La:"#70D4FF", Ce:"#FFFFC7", Pr:"#D9FFC7", Nd:"#C7FFC7",
  Pm:"#A3FFC7", Sm:"#8FFFC7", Eu:"#61FFC7", Gd:"#45FFC7", Tb:"#30FFC7",
  Dy:"#1FFFC7", Ho:"#00FF9C", Er:"#00E675", Tm:"#00D452", Yb:"#00BF38",
  Lu:"#00AB24", Hf:"#4DC2FF", Ta:"#4DA6FF", W:"#2194D6", Re:"#267DAB",
  Os:"#266696", Ir:"#175487", Pt:"#D0D0E0", Au:"#FFD123", Hg:"#B8B8D0",
  Tl:"#A6544D", Pb:"#575961", Bi:"#9E4FB5", Po:"#AB5C00", At:"#754F45",
  Rn:"#428296", Fr:"#420066", Ra:"#007D00", Ac:"#70ABFA", Th:"#00BAFF",
  Pa:"#00A1FF", U:"#008FFF", Np:"#0080FF", Pu:"#006BFF", Am:"#545CF2",
  Cm:"#785CE3", Bk:"#8A4FE3", Cf:"#A136D4", Es:"#B31FD4", Fm:"#B31FBA",
  Md:"#B30DA6", No:"#BD0D87", Lr:"#C70066", Rf:"#CC0059", Db:"#D1004F",
  Sg:"#D90045", Bh:"#E00038", Hs:"#E6002E", Mt:"#EB0026",
};
const FALLBACK_COLOR = "#808080";
let BOND_COLOR = "#d3d3d3";
const colorOverrides = Object.create(null);

export function elementColor(symbol) {
  return colorOverrides[symbol] || CPK_COLORS[symbol] || FALLBACK_COLOR;
}

export function setGlobalElementColor(symbol, hex) {
  colorOverrides[symbol] = hex;
}

export function setGlobalBondColor(hex) {
  BOND_COLOR = hex;
}

export function bondColor() {
  return BOND_COLOR;
}

function cross(u, v) {
  return [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
}
function sub(u, v) { return [u[0]-v[0], u[1]-v[1], u[2]-v[2]]; }
function add(u, v) { return [u[0]+v[0], u[1]+v[1], u[2]+v[2]]; }
function scale(u, s) { return [u[0]*s, u[1]*s, u[2]*s]; }
function dot(u, v) { return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]; }
function norm(u) {
  const n = Math.hypot(u[0], u[1], u[2]);
  return n < 1e-12 ? null : scale(u, 1 / n);
}

/** @returns {number[]|null} unit normal or null if invalid */
export function millerNormal(cell, h, k, l) {
  if (h === 0 && k === 0 && l === 0) return null;
  const a = cell[0], b = cell[1], c = cell[2];
  const aS = cross(b, c), bS = cross(c, a), cS = cross(a, b);
  return norm(add(add(scale(aS, h), scale(bS, k)), scale(cS, l)));
}

function cellCenter(cell) {
  const a = cell[0], b = cell[1], c = cell[2];
  const corners = [
    [0,0,0], a, b, c,
    add(a,b), add(a,c), add(b,c), add(add(a,b),c),
  ];
  return scale(corners.reduce((s, p) => add(s, p), [0,0,0]), 1/8);
}

export function planeOffsetFromCenter(cell, normal, depthFrac) {
  const a = cell[0], b = cell[1], c = cell[2];
  const corners = [
    [0,0,0], a, b, c,
    add(a,b), add(a,c), add(b,c), add(add(a,b),c),
  ];
  const center = cellCenter(cell);
  const projs = corners.map((p) => dot(sub(p, center), normal));
  const half = 0.5 * (Math.max(...projs) - Math.min(...projs));
  const frac = Math.max(-1, Math.min(1, depthFrac));
  return scale(normal, frac * half);
}

export function planeSize(cell) {
  const a = cell[0], b = cell[1], c = cell[2];
  const diags = [
    Math.hypot(...add(a,b)),
    Math.hypot(...add(a,c)),
    Math.hypot(...add(b,c)),
  ];
  return 1.2 * Math.max(...diags);
}

export class CrystalViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#1e1e24");

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    this.camera.position.set(0, 0, 30);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(1, 1, 1);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-1, -0.5, -1);
    this.scene.add(fill);

    this.content = new THREE.Group();
    this.scene.add(this.content);

    this.atomScale = 0.5;
    this.bondRadius = 0.1;
    this.pbrShiny = false;
    this.showAxes = true;
    this._axes = new THREE.AxesHelper(5);
    this.scene.add(this._axes);

    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this._tooltip = document.createElement("div");
    this._tooltip.className = "crystal-atom-tooltip";
    this._tooltip.style.cssText = "position:absolute;pointer-events:none;display:none;padding:4px 8px;background:#111;color:#eee;font:12px/1.3 sans-serif;border-radius:4px;z-index:5;";
    container.style.position = container.style.position || "relative";
    container.appendChild(this._tooltip);
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this.renderer.domElement.addEventListener("pointerdown", this._onPointerDown);
    this.renderer.domElement.addEventListener("pointermove", this._onPointerMove);

    this._atomIndexByMesh = new WeakMap();
    this.geometry = null;

    this.eraserEnabled = false;
    this._erasedAtomIndices = new Set();

    this._cut = { h: 0, k: 0, l: 1, depthFrac: 0, color: "#00ffff", visible: false };
    this._cutMesh = null;

    this._resize = this._resize.bind(this);
    window.addEventListener("resize", this._resize);
    if (window.ResizeObserver) {
      new ResizeObserver(this._resize).observe(container);
    }
    this._resize();
    this._animate();
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    const aspect = w / h;
    if (this.camera.isOrthographicCamera) {
      const half = this._orthoHalfHeight ?? Math.max((this.camera.top - this.camera.bottom) / 2, 1);
      this._orthoHalfHeight = half;
      this.camera.left = -half * aspect;
      this.camera.right = half * aspect;
      this.camera.top = half;
      this.camera.bottom = -half;
      this.camera.updateProjectionMatrix();
    } else {
      this.camera.aspect = aspect;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.setSize(w, h, false);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  clear() {
    this.content.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    this.content.clear();
    this._cutMesh = null;
  }

  /* Draws a payload from POST /api/crystal/{name}/geometry. */
  render(geometry, options = {}) {
    const { showBonds = true, showPolyhedra = false, showCell = true, frame = true } = options;
    this.geometry = geometry;
    this._lastOptions = options;
    this.clear();

    const center = new THREE.Vector3(...geometry.center);

    this._drawAtoms(geometry, center);
    if (showBonds) this._drawBonds(geometry, center);
    if (showPolyhedra && geometry.polyhedra?.length) this._drawPolyhedra(geometry, center);
    const drawCell = showCell && geometry.show_cell !== false;
    if (drawCell) this._drawCell(geometry, center);

    if (this._moireEnvelope) this._drawMoire(this._moireEnvelope, center);
    if (this._bzData) this._drawBZ(this._bzData, center);

    this._syncCutPlane();
    if (frame) this.frameToContent();
  }

  /* Brillouin zone as a separate mesh group from the atom InstancedMeshes. */
  setBrillouinZone(bz, { frame = true } = {}) {
    this._bzData = bz;
    if (!this.geometry) {
      this.clear();
      const center = new THREE.Vector3(0, 0, 0);
      this._drawBZ(bz, center);
      if (frame) this.frameToContent();
      return;
    }
    this.render(this.geometry, { ...(this._lastOptions || {}), frame });
  }

  clearBrillouinZone() {
    this._bzData = null;
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  setMoireEnvelope(points, { frame = false } = {}) {
    this._moireEnvelope = points;
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame });
  }

  clearMoire() {
    this._moireEnvelope = null;
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  _drawBZ(bz, center) {
    const style = bz.style || "solid";
    const points = bz.hull_points.map((p) => new THREE.Vector3(...p).sub(center));

    if (style === "solid" || style === "both") {
      const positions = [];
      const indices = [];
      points.forEach((p) => positions.push(p.x, p.y, p.z));
      bz.simplices.forEach((tri) => indices.push(...tri));

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();

      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: "#ff00ff",
          transparent: true,
          opacity: 0.25,
          side: THREE.DoubleSide,
          depthWrite: false,
          roughness: 0.6,
          metalness: 0.05,
        })
      );
      this.content.add(mesh);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: "#ffffff" })
      );
      this.content.add(edges);
    }

    if (style === "skeleton" || style === "both") {
      const linePoints = [];
      (bz.edges || []).forEach(([a, b]) => {
        linePoints.push(new THREE.Vector3(...a).sub(center), new THREE.Vector3(...b).sub(center));
      });
      if (linePoints.length) {
        this.content.add(new THREE.LineSegments(
          new THREE.BufferGeometry().setFromPoints(linePoints),
          new THREE.LineBasicMaterial({ color: "#ff00ff", linewidth: 2 })
        ));
      }
    }

    // Reciprocal axes from the origin.
    const axisLen = Math.max(...points.map((p) => p.length()), 1) * 1.25;
    [
      [[axisLen, 0, 0], "#ff6666", "kx"],
      [[0, axisLen, 0], "#66ff66", "ky"],
      [[0, 0, axisLen], "#6666ff", "kz"],
    ].forEach(([dir]) => {
      const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(...dir).normalize(),
        new THREE.Vector3(0, 0, 0).sub(center),
        axisLen,
        0xffffff,
        0.12 * axisLen,
        0.06 * axisLen
      );
      this.content.add(arrow);
    });

    if (bz.surface_vertices && bz.surface_simplices) {
      const positions = [];
      bz.surface_vertices.forEach((p) => {
        const v = new THREE.Vector3(...p).sub(center);
        positions.push(v.x, v.y, v.z);
      });
      const indices = [];
      bz.surface_simplices.forEach((tri) => indices.push(...tri));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      this.content.add(new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: "#00ffff",
          transparent: true,
          opacity: 0.35,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
      ));
    }

    if (bz.projection_lines && bz.projection_lines.length) {
      const linePoints = [];
      bz.projection_lines.forEach(([a, b]) => {
        linePoints.push(new THREE.Vector3(...a).sub(center), new THREE.Vector3(...b).sub(center));
      });
      this.content.add(new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(linePoints),
        new THREE.LineBasicMaterial({ color: "#ffd700" })
      ));
    }
  }

  _drawMoire(points, center) {
    if (!points || points.length < 2) return;
    const linePoints = points.map((p) => new THREE.Vector3(...p).sub(center));
    this.content.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(linePoints),
      new THREE.LineBasicMaterial({ color: "#ffd700" })
    ));
  }

  setEraserEnabled(on) {
    this.eraserEnabled = Boolean(on);
    this.controls.enabled = !this.eraserEnabled;
    if (!this.eraserEnabled) this._tooltip.style.display = "none";
  }

  getOmittedAtomIndices() {
    return [...this._erasedAtomIndices].sort((a, b) => a - b);
  }

  clearErasedAtoms() {
    if (this._erasedAtomIndices.size === 0) return;
    this._erasedAtomIndices.clear();
    if (this.geometry) {
      this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
    }
  }

  omitAtom(index) {
    if (this._erasedAtomIndices.has(index)) return;
    this._erasedAtomIndices.add(index);
    if (this.geometry) {
      this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
    }
  }

  /* One InstancedMesh per element keeps thousands of atoms at one draw call each. */
  _drawAtoms(geometry, center) {
    const byElement = new Map();
    geometry.atoms.forEach((atom, index) => {
      if (this._erasedAtomIndices.has(index)) return;
      if (!byElement.has(atom.element)) byElement.set(atom.element, []);
      byElement.get(atom.element).push({ atom, index });
    });

    const sphere = new THREE.SphereGeometry(1, 20, 16);
    const dummy = new THREE.Object3D();

    for (const [element, entries] of byElement) {
      const material = new THREE.MeshStandardMaterial({
        color: elementColor(element),
        ...pbrMaterialParams(this.pbrShiny, "atom"),
      });
      const mesh = new THREE.InstancedMesh(sphere, material, entries.length);

      entries.forEach(({ atom }, slot) => {
        dummy.position.set(...atom.position).sub(center);
        const r = atom.radius * this.atomScale;
        dummy.scale.set(r, r, r);
        dummy.updateMatrix();
        mesh.setMatrixAt(slot, dummy.matrix);
      });

      mesh.instanceMatrix.needsUpdate = true;
      this._atomIndexByMesh.set(mesh, entries.map((e) => e.index));
      this.content.add(mesh);
    }
  }

  _raycastAtomIndex(event) {
    if (!this.geometry) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._pointer, this.camera);
    const meshes = [];
    this.content.traverse((o) => { if (o.isInstancedMesh && this._atomIndexByMesh.has(o)) meshes.push(o); });
    const hits = this._raycaster.intersectObjects(meshes, false);
    if (!hits.length) return null;
    const hit = hits[0];
    const indices = this._atomIndexByMesh.get(hit.object);
    return indices[hit.instanceId];
  }

  _onPointerDown(event) {
    if (!this.eraserEnabled) return;
    const index = this._raycastAtomIndex(event);
    if (index !== null) this.omitAtom(index);
  }

  _onPointerMove(event) {
    if (!this.geometry) { this._tooltip.style.display = "none"; return; }
    if (this.eraserEnabled) {
      this._tooltip.style.display = "none";
      if (event.buttons > 0) {
        const index = this._raycastAtomIndex(event);
        if (index !== null) this.omitAtom(index);
      }
      return;
    }
    const index = this._raycastAtomIndex(event);
    if (index === null) { this._tooltip.style.display = "none"; return; }
    const atom = this.geometry.atoms[index];
    if (!atom) { this._tooltip.style.display = "none"; return; }
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._tooltip.textContent = `${atom.label} (${atom.element})`;
    this._tooltip.style.display = "block";
    this._tooltip.style.left = `${event.clientX - rect.left + 12}px`;
    this._tooltip.style.top = `${event.clientY - rect.top + 12}px`;
  }

  /* Unit cylinders on +Y, oriented per bond, so all bonds share one geometry. */
  _drawBonds(geometry, center) {
    const bonds = geometry.bonds.filter(
      (bond) => !this._erasedAtomIndices.has(bond.i) && !this._erasedAtomIndices.has(bond.j),
    );
    if (!bonds.length) return;

    const cylinder = new THREE.CylinderGeometry(this.bondRadius, this.bondRadius, 1, 8);
    const material = new THREE.MeshStandardMaterial({
      color: BOND_COLOR,
      ...pbrMaterialParams(this.pbrShiny, "bond"),
    });
    const mesh = new THREE.InstancedMesh(cylinder, material, bonds.length);

    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const start = new THREE.Vector3();
    const end = new THREE.Vector3();
    const direction = new THREE.Vector3();

    bonds.forEach((bond, slot) => {
      start.set(...geometry.atoms[bond.i].position).sub(center);
      end.set(...geometry.atoms[bond.j].position).sub(center);
      direction.subVectors(end, start);
      const length = direction.length();

      dummy.position.copy(start).addScaledVector(direction, 0.5);
      dummy.quaternion.setFromUnitVectors(up, direction.normalize());
      dummy.scale.set(1, length, 1);
      dummy.updateMatrix();
      mesh.setMatrixAt(slot, dummy.matrix);
    });

    mesh.instanceMatrix.needsUpdate = true;
    this.content.add(mesh);
  }

  _drawPolyhedra(geometry, center) {
    for (const poly of geometry.polyhedra) {
      if (this._erasedAtomIndices.has(poly.center)) continue;
      if (poly.vertex_atom_indices?.some((idx) => this._erasedAtomIndices.has(idx))) continue;

      const centerAtom = geometry.atoms[poly.center];
      if (!centerAtom || !poly.vertices?.length || !poly.simplices?.length) continue;

      const positions = [];
      poly.vertices.forEach((p) => {
        const v = new THREE.Vector3(...p).sub(center);
        positions.push(v.x, v.y, v.z);
      });
      const indices = [];
      poly.simplices.forEach((tri) => indices.push(...tri));

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geom.setIndex(indices);
      geom.computeVertexNormals();

      this.content.add(new THREE.Mesh(
        geom,
        new THREE.MeshStandardMaterial({
          color: elementColor(centerAtom.element),
          transparent: true,
          opacity: 0.3,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      ));
    }
  }

  _drawCell(geometry, center) {
    const [a, b, c] = geometry.cell.map((row) => new THREE.Vector3(...row));
    const origin = new THREE.Vector3(0, 0, 0);

    const corners = [
      origin,
      a,
      b,
      c,
      new THREE.Vector3().addVectors(a, b),
      new THREE.Vector3().addVectors(a, c),
      new THREE.Vector3().addVectors(b, c),
      new THREE.Vector3().addVectors(a, b).add(c),
    ];
    const edges = [
      [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4],
      [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
    ];

    const points = [];
    edges.forEach(([s, e]) => {
      points.push(corners[s].clone().sub(center), corners[e].clone().sub(center));
    });

    const lines = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: "#6f7891" })
    );
    this.content.add(lines);
  }

  /* Pulls the camera back far enough that the whole cell fits the viewport. */
  frameToContent() {
    const box = new THREE.Box3().setFromObject(this.content);
    if (box.isEmpty()) return;

    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 1);
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    const aspect = w / h;

    if (this.camera.isOrthographicCamera) {
      const half = radius * 1.15;
      this._orthoHalfHeight = half;
      this.camera.left = -half * aspect;
      this.camera.right = half * aspect;
      this.camera.top = half;
      this.camera.bottom = -half;
      this.camera.position.set(0, 0, 30);
      this.camera.near = 0.1;
      this.camera.far = 5000;
      this.camera.updateProjectionMatrix();
    } else {
      const distance = radius / Math.sin((this.camera.fov * Math.PI) / 360);
      this.camera.position.set(0, 0, distance * 1.15);
      this.camera.near = distance / 100;
      this.camera.far = distance * 100;
      this.camera.aspect = aspect;
      this.camera.updateProjectionMatrix();
    }
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  /* Redraws in place: the camera stays where the user left it. */
  setAtomScale(scale) {
    this.atomScale = scale;
    if (!this.geometry) return;
    this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  setBondRadius(r) {
    this.bondRadius = r;
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  setPbrShiny(on) {
    this.pbrShiny = Boolean(on);
    if (this.geometry) {
      this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
    }
  }

  setShowAxes(on) {
    this.showAxes = on;
    this._axes.visible = on;
  }

  setProjection(mode) {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    const aspect = w / h;
    const dist = this.camera.position.length();
    if (mode === "orthographic") {
      const half = this.camera.isOrthographicCamera
        ? Math.max((this.camera.top - this.camera.bottom) / 2, 1)
        : Math.max(dist * Math.tan((45 * Math.PI) / 360), 1);
      this._orthoHalfHeight = half;
      const cam = new THREE.OrthographicCamera(-half * aspect, half * aspect, half, -half, 0.1, 5000);
      cam.position.copy(this.camera.position);
      cam.quaternion.copy(this.camera.quaternion);
      this.camera = cam;
    } else {
      const cam = new THREE.PerspectiveCamera(45, aspect, 0.1, 5000);
      cam.position.copy(this.camera.position);
      cam.quaternion.copy(this.camera.quaternion);
      this.camera = cam;
      this._orthoHalfHeight = undefined;
    }
    this.controls.object = this.camera;
    this._resize();
  }

  lookAlong(which) {
    if (!this.geometry) return;
    const [a, b, c] = this.geometry.cell.map((row) => new THREE.Vector3(...row));
    let dir;
    if (which === "+a") dir = a.clone();
    else if (which === "+b") dir = b.clone();
    else if (which === "+c") dir = c.clone();
    else dir = a.clone().add(b).add(c);
    dir.normalize();
    const dist = this.camera.position.length() || 30;
    this.camera.position.copy(dir.multiplyScalar(dist));
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  lookAlongMiller(h, k, l) {
    if (!this.geometry) return;
    const n = millerNormal(this.geometry.cell, h, k, l);
    if (!n || (n[0] === 0 && n[1] === 0 && n[2] === 0)) return;
    const dir = new THREE.Vector3(...n).normalize();
    const dist = this.camera.position.length() || 30;
    this.camera.position.copy(dir.multiplyScalar(dist));
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setAzEl(azDeg, elDeg) {
    const dist = this.camera.position.length() || 30;
    const az = (azDeg * Math.PI) / 180;
    const el = (elDeg * Math.PI) / 180;
    this.camera.position.set(
      dist * Math.cos(el) * Math.sin(az),
      dist * Math.sin(el),
      dist * Math.cos(el) * Math.cos(az),
    );
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setElementColor(symbol, hex) {
    setGlobalElementColor(symbol, hex);
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  setBondColor(hex) {
    setGlobalBondColor(hex);
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  legend() {
    if (!this.geometry) return [];
    return this.geometry.elements.map((el) => ({ element: el, color: elementColor(el) }));
  }

  _syncCutPlane() {
    if (this._cutMesh) {
      this.content.remove(this._cutMesh);
      this._cutMesh.geometry?.dispose();
      this._cutMesh.material?.dispose();
      this._cutMesh = null;
    }
    if (!this._cut.visible || !this.geometry) return;
    const { h, k, l, depthFrac, color } = this._cut;
    const n = millerNormal(this.geometry.cell, h, k, l);
    if (!n) return;
    const offset = planeOffsetFromCenter(this.geometry.cell, n, depthFrac);
    const size = planeSize(this.geometry.cell);
    const geom = new THREE.PlaneGeometry(size, size);
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    const zAxis = new THREE.Vector3(0, 0, 1);
    const normal = new THREE.Vector3(...n);
    if (normal.dot(zAxis) < -0.999999) {
      mesh.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    } else {
      mesh.quaternion.setFromUnitVectors(zAxis, normal);
    }
    const pos = sub(add(cellCenter(this.geometry.cell), offset), this.geometry.center);
    mesh.position.set(...pos);
    this._cutMesh = mesh;
    this.content.add(mesh);
  }

  setCutPlane({ h, k, l, depthFrac, color, visible } = {}) {
    if (h !== undefined) this._cut.h = Number(h) || 0;
    if (k !== undefined) this._cut.k = Number(k) || 0;
    if (l !== undefined) this._cut.l = Number(l) || 0;
    if (depthFrac !== undefined) this._cut.depthFrac = Number(depthFrac) || 0;
    if (color !== undefined) this._cut.color = color;
    if (visible !== undefined) this._cut.visible = Boolean(visible);
    this._syncCutPlane();
  }

  capturePNG(scale = 2) {
    const canvas = this.renderer.domElement;
    const w = canvas.width;
    const h = canvas.height;
    const prevPR = this.renderer.getPixelRatio();
    this.renderer.setPixelRatio(scale);
    this._resize();
    this.renderer.render(this.scene, this.camera);
    const url = this.renderer.domElement.toDataURL("image/png");
    this.renderer.setPixelRatio(prevPR);
    this._resize();
    this.renderer.render(this.scene, this.camera);
    return url;
  }
}
