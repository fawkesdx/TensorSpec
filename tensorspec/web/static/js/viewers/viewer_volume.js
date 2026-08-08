/* ARPES BZ-prism volume viewer.
 *
 * Extrudes a rectangular or hexagonal footprint into a 3D prism, paints
 * intensity on the walls / Fermi plane / indentation faces from a downsampled
 * volume. Geometry math stays on the server (prism polygon); this module draws.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { COLORMAPS } from "/static/js/viewers/colormaps.js";

function sampleVolume(values, shape, ix, iy, iz) {
    const [nz, ny, nx] = shape;
    const x = Math.max(0, Math.min(nx - 1, ix));
    const y = Math.max(0, Math.min(ny - 1, iy));
    const z = Math.max(0, Math.min(nz - 1, iz));
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const z0 = Math.floor(z);
    const x1 = Math.min(nx - 1, x0 + 1);
    const y1 = Math.min(ny - 1, y0 + 1);
    const z1 = Math.min(nz - 1, z0 + 1);
    const xd = x - x0;
    const yd = y - y0;
    const zd = z - z0;
    const at = (zi, yi, xi) => values[(zi * ny + yi) * nx + xi];
    const c00 = at(z0, y0, x0) * (1 - xd) + at(z0, y0, x1) * xd;
    const c01 = at(z0, y1, x0) * (1 - xd) + at(z0, y1, x1) * xd;
    const c10 = at(z1, y0, x0) * (1 - xd) + at(z1, y0, x1) * xd;
    const c11 = at(z1, y1, x0) * (1 - xd) + at(z1, y1, x1) * xd;
    const c0 = c00 * (1 - yd) + c01 * yd;
    const c1 = c10 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
}

function worldToIndex(axis, value) {
    if (!axis.length) return 0;
    if (axis.length === 1) return 0;
    const t = (value - axis[0]) / (axis[axis.length - 1] - axis[0] || 1);
    return t * (axis.length - 1);
}

function colorize(level, table) {
    const entry = Math.max(0, Math.min(255, Math.round(level * 255))) * 3;
    return [table[entry], table[entry + 1], table[entry + 2]];
}

export class ArpesVolumeViewer {
    constructor(container) {
        this.container = container;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color("#12121a");
        this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
        this.camera.position.set(2.5, 2.2, 2.8);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.65));
        const key = new THREE.DirectionalLight(0xffffff, 0.85);
        key.position.set(1, 1.4, 0.8);
        this.scene.add(key);

        this.root = new THREE.Group();
        this.scene.add(this.root);
        this.header = null;
        this.values = null;
        this.options = {
            indentSectors: 1,
            indentDepth: 0.55,
            showFermi: true,
            eFermi: 0,
            opacity: 0.92,
            colormap: "magma",
        };

        this._resize = this._resize.bind(this);
        window.addEventListener("resize", this._resize);
        if (window.ResizeObserver) new ResizeObserver(this._resize).observe(container);
        this._resize();
        this._animate();
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    _resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (!w || !h) return;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h, false);
    }

    clear() {
        while (this.root.children.length) {
            const obj = this.root.children.pop();
            obj.traverse((o) => {
                if (o.geometry) o.geometry.dispose();
                if (o.material) {
                    if (o.material.map) o.material.map.dispose();
                    o.material.dispose();
                }
            });
        }
    }

    setVolume(header, values, options = {}) {
        this.header = header;
        this.values = values;
        this.options = { ...this.options, ...options };
        this.rebuild();
    }

    setOptions(options = {}) {
        this.options = { ...this.options, ...options };
        if (this.header && this.values) this.rebuild();
    }

    rebuild() {
        if (!this.header || !this.values) return;
        this.clear();
        const prism = this.header.prism;
        const kx = prism.kx.slice(0, -1); // drop closing point
        const ky = prism.ky.slice(0, -1);
        const n = kx.length;
        if (n < 3) return;

        const zAxis = this.header.z_axis;
        const zMin = zAxis[0];
        const zMax = zAxis[zAxis.length - 1];
        const cx = kx.reduce((a, b) => a + b, 0) / n;
        const cy = ky.reduce((a, b) => a + b, 0) / n;

        const indentN = Math.max(0, Math.min(n - 1, Number(this.options.indentSectors) || 0));
        const depth = Math.max(0, Math.min(1, Number(this.options.indentDepth) || 0));
        // Vertices kept on the outer wall; indent removes [0 .. indentN) edges.
        const cutStart = 0;
        const cutEnd = indentN;

        // Outer side walls (skip indented sector)
        for (let i = 0; i < n; i++) {
            if (indentN > 0 && i >= cutStart && i < cutEnd) continue;
            const j = (i + 1) % n;
            this._addWall(
                { x: kx[i], y: ky[i] },
                { x: kx[j], y: ky[j] },
                zMin,
                zMax
            );
        }

        // Indentation radial faces + floor of the notch
        if (indentN > 0 && depth > 0) {
            const i0 = cutStart;
            const i1 = cutEnd % n;
            const p0 = { x: kx[i0], y: ky[i0] };
            const p1 = { x: kx[i1], y: ky[i1] };
            const inner0 = {
                x: cx + (p0.x - cx) * (1 - depth),
                y: cy + (p0.y - cy) * (1 - depth),
            };
            const inner1 = {
                x: cx + (p1.x - cx) * (1 - depth),
                y: cy + (p1.y - cy) * (1 - depth),
            };
            this._addWall(p0, inner0, zMin, zMax);
            this._addWall(inner1, p1, zMin, zMax);
            // Inner arc approximating the notch back wall along remaining cut vertices
            if (indentN >= 1) {
                this._addWall(inner0, inner1, zMin, zMax);
            }
            // Optional: paint exposed outer chord on indented region at reduced radius only
        }

        // Top lid (full polygon, slightly translucent)
        this._addPolygonLid(kx, ky, zMax, 0.35);

        // Fermi / energy slice plane inside the prism
        if (this.options.showFermi) {
            const eF = Number(this.options.eFermi);
            const zF = Math.max(zMin, Math.min(zMax, eF));
            this._addPolygonLid(kx, ky, zF, 0.85, true);
        }

        // Frame camera once
        const box = new THREE.Box3().setFromObject(this.root);
        if (!box.isEmpty()) {
            const size = new THREE.Vector3();
            const center = new THREE.Vector3();
            box.getSize(size);
            box.getCenter(center);
            const span = Math.max(size.x, size.y, size.z, 1);
            this.controls.target.copy(center);
            this.camera.position.set(center.x + span * 1.4, center.y + span * 1.1, center.z + span * 1.5);
            this.controls.update();
        }
    }

    _addWall(a, b, z0, z1) {
        const width = 96;
        const height = 96;
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        const img = ctx.createImageData(width, height);
        const table = COLORMAPS[this.options.colormap] || COLORMAPS.magma;
        const { vmin, vmax, shape, x_axis: xAxis, y_axis: yAxis, z_axis: zAxis } = this.header;
        const span = vmax - vmin || 1;

        for (let row = 0; row < height; row++) {
            const fz = row / (height - 1 || 1);
            const z = z0 + (z1 - z0) * fz;
            const iz = worldToIndex(zAxis, z);
            for (let col = 0; col < width; col++) {
                const ft = col / (width - 1 || 1);
                const x = a.x + (b.x - a.x) * ft;
                const y = a.y + (b.y - a.y) * ft;
                const ix = worldToIndex(xAxis, x);
                const iy = worldToIndex(yAxis, y);
                const v = sampleVolume(this.values, shape, ix, iy, iz);
                const [r, g, bcol] = colorize((v - vmin) / span, table);
                const o = (row * width + col) * 4;
                img.data[o] = r;
                img.data[o + 1] = g;
                img.data[o + 2] = bcol;
                img.data[o + 3] = Math.round(255 * (this.options.opacity ?? 0.92));
            }
        }
        ctx.putImageData(img, 0, 0);
        const tex = new THREE.CanvasTexture(canvas);
        tex.colorSpace = THREE.SRGBColorSpace;

        const positions = new Float32Array([
            a.x, a.y, z0,
            b.x, b.y, z0,
            b.x, b.y, z1,
            a.x, a.y, z1,
        ]);
        // Map: local X along wall, Y along energy (canvas row 0 = z0 at bottom of texture → flip V)
        const uvs = new Float32Array([
            0, 0,
            1, 0,
            1, 1,
            0, 1,
        ]);
        const idx = new Uint16Array([0, 1, 2, 0, 2, 3]);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
        geo.setIndex(new THREE.BufferAttribute(idx, 1));
        geo.computeVertexNormals();
        const mat = new THREE.MeshBasicMaterial({
            map: tex,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: true,
        });
        this.root.add(new THREE.Mesh(geo, mat));
    }

    _addPolygonLid(kx, ky, z, opacity = 0.8, sampleIntensity = false) {
        const n = kx.length;
        if (n < 3) return;
        const shape = new THREE.Shape();
        shape.moveTo(kx[0], ky[0]);
        for (let i = 1; i < n; i++) shape.lineTo(kx[i], ky[i]);
        shape.closePath();
        const geo = new THREE.ShapeGeometry(shape);
        // Lift to z
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
            pos.setZ(i, z);
        }
        pos.needsUpdate = true;
        geo.computeVertexNormals();

        let material;
        if (sampleIntensity) {
            const size = 128;
            const canvas = document.createElement("canvas");
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext("2d");
            const img = ctx.createImageData(size, size);
            const table = COLORMAPS[this.options.colormap] || COLORMAPS.magma;
            const { vmin, vmax, shape: volShape, x_axis: xAxis, y_axis: yAxis, z_axis: zAxis } = this.header;
            const span = vmax - vmin || 1;
            const iz = worldToIndex(zAxis, z);
            const x0 = Math.min(...kx);
            const x1 = Math.max(...kx);
            const y0 = Math.min(...ky);
            const y1 = Math.max(...ky);
            for (let row = 0; row < size; row++) {
                const y = y0 + ((size - 1 - row) / (size - 1 || 1)) * (y1 - y0);
                for (let col = 0; col < size; col++) {
                    const x = x0 + (col / (size - 1 || 1)) * (x1 - x0);
                    const ix = worldToIndex(xAxis, x);
                    const iy = worldToIndex(yAxis, y);
                    const v = sampleVolume(this.values, volShape, ix, iy, iz);
                    const [r, g, b] = colorize((v - vmin) / span, table);
                    const o = (row * size + col) * 4;
                    img.data[o] = r;
                    img.data[o + 1] = g;
                    img.data[o + 2] = b;
                    img.data[o + 3] = Math.round(255 * opacity);
                }
            }
            ctx.putImageData(img, 0, 0);
            const tex = new THREE.CanvasTexture(canvas);
            tex.colorSpace = THREE.SRGBColorSpace;
            // Remap UVs of ShapeGeometry to bounding box
            const uv = geo.attributes.uv;
            for (let i = 0; i < pos.count; i++) {
                const px = pos.getX(i);
                const py = pos.getY(i);
                uv.setXY(i, (px - x0) / (x1 - x0 || 1), (py - y0) / (y1 - y0 || 1));
            }
            uv.needsUpdate = true;
            material = new THREE.MeshBasicMaterial({
                map: tex,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: false,
            });
        } else {
            material = new THREE.MeshBasicMaterial({
                color: 0x334155,
                transparent: true,
                opacity,
                side: THREE.DoubleSide,
                depthWrite: false,
            });
        }
        this.root.add(new THREE.Mesh(geo, material));

        // Outline
        const pts = kx.map((x, i) => new THREE.Vector3(x, ky[i], z));
        pts.push(pts[0].clone());
        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: 0x22d3ee })
        );
        this.root.add(line);
    }
}
