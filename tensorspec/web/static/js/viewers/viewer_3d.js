/* Crystal viewport: renders the geometry payload from /api/crystal.
 *
 * This module draws only. Every position, radius and bond pair is decided by
 * `core/crystallography.py`; nothing here computes geometry. Colours are the
 * one exception, being a display choice rather than a property of the crystal.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const CPK_COLORS = {
  H: "#FFFFFF", C: "#333333", N: "#2233FF", O: "#FF2200",
  Te: "#FF8C00", Fe: "#E06633", Ta: "#B041FF", Ir: "#0080FF",
  Nb: "#7A378B", W: "#4682B4", Mo: "#5F9EA0",
};
const FALLBACK_COLOR = "#008080";
const BOND_COLOR = "#d3d3d3";

export function elementColor(symbol) {
  return CPK_COLORS[symbol] || FALLBACK_COLOR;
}

export class CrystalViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#1e1e24");

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    this.camera.position.set(0, 0, 30);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
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
    this.geometry = null;

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
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
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
  }

  /* Draws a payload from POST /api/crystal/{name}/geometry. */
  render(geometry, options = {}) {
    const { showBonds = true, showCell = true, frame = true } = options;
    this.geometry = geometry;
    this._lastOptions = options;
    this.clear();

    const center = new THREE.Vector3(...geometry.center);

    this._drawAtoms(geometry, center);
    if (showBonds) this._drawBonds(geometry, center);
    if (showCell) this._drawCell(geometry, center);

    if (frame) this.frameToContent();
  }

  /* One InstancedMesh per element keeps thousands of atoms at one draw call each. */
  _drawAtoms(geometry, center) {
    const byElement = new Map();
    geometry.atoms.forEach((atom, index) => {
      if (!byElement.has(atom.element)) byElement.set(atom.element, []);
      byElement.get(atom.element).push({ atom, index });
    });

    const sphere = new THREE.SphereGeometry(1, 20, 16);
    const dummy = new THREE.Object3D();

    for (const [element, entries] of byElement) {
      const material = new THREE.MeshStandardMaterial({
        color: elementColor(element),
        roughness: 0.45,
        metalness: 0.1,
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
      this.content.add(mesh);
    }
  }

  /* Unit cylinders on +Y, oriented per bond, so all bonds share one geometry. */
  _drawBonds(geometry, center) {
    if (!geometry.bonds.length) return;

    const cylinder = new THREE.CylinderGeometry(this.bondRadius, this.bondRadius, 1, 8);
    const material = new THREE.MeshStandardMaterial({
      color: BOND_COLOR,
      roughness: 0.5,
      metalness: 0.1,
    });
    const mesh = new THREE.InstancedMesh(cylinder, material, geometry.bonds.length);

    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const start = new THREE.Vector3();
    const end = new THREE.Vector3();
    const direction = new THREE.Vector3();

    geometry.bonds.forEach((bond, slot) => {
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
    const distance = radius / Math.sin((this.camera.fov * Math.PI) / 360);

    this.camera.position.set(0, 0, distance * 1.15);
    this.camera.near = distance / 100;
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  /* Redraws in place: the camera stays where the user left it. */
  setAtomScale(scale) {
    this.atomScale = scale;
    if (!this.geometry) return;
    this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  legend() {
    if (!this.geometry) return [];
    return this.geometry.elements.map((el) => ({ element: el, color: elementColor(el) }));
  }
}
