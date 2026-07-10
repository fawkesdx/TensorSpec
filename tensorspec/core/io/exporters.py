# File: tensorspec/core/io/exporters.py

class SceneExporter:
    """Handles formatting and writing 3D crystal data to external DCC software scripts."""

    @staticmethod
    def export_3dsmax(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data=None):
        atoms_formatted = ",\n        ".join(str(a) for a in atoms_data)
        bonds_formatted = ",\n        ".join(str(b) for b in bonds_data)
        lattice_formatted = ",\n        ".join(str(l) for l in lattice_data)

        bz_solid_script = ""
        if bz_solid_data:
            verts = bz_solid_data['verts']
            faces = bz_solid_data['faces']
            verts_str = ", ".join([f"[{p[0]},{p[1]},{p[2]}]" for p in verts])
            faces_str = ", ".join([f"[{f[0]+1},{f[1]+1},{f[2]+1}]" for f in faces]) 
            bz_solid_script = f'''
    bz_mesh = rt.mesh(vertices=rt.Array({verts_str}), faces=rt.Array({faces_str}))
    bz_mesh.material = get_material("#FF00FF")
    bz_mesh.material.opacity = 0.3
'''

        script_content = f'''import pymxs
rt = pymxs.runtime

def hex_to_color(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def get_material(hex_color):
    mat_name = "Mat_" + hex_color.replace('#', '')
    mat = rt.getNodeByName(mat_name)
    if not mat:
        mat = rt.PhysicalMaterial()
        mat.name = mat_name
        r, g, b = hex_to_color(hex_color)
        mat.Base_Color = rt.Color(r, g, b)
        mat.roughness = 0.1
    return mat

with pymxs.undo(True, 'Create Crystal'):
    base_atoms = {{}}
    for ax, ay, az, rad, color in [{atoms_formatted}]:
        key = (color, rad)
        if key not in base_atoms:
            sphere = rt.Sphere(radius=rad, segs=32)
            sphere.material = get_material(color)
            sphere.pos = rt.Point3(ax, ay, az)
            base_atoms[key] = sphere
        else:
            inst = rt.instance(base_atoms[key])
            inst.pos = rt.Point3(ax, ay, az)

    for x1, y1, z1, x2, y2, z2, rad, color in [{bonds_formatted}]:
        p1, p2 = rt.Point3(x1, y1, z1), rt.Point3(x2, y2, z2)
        cyl = rt.Cylinder(radius=rad, height=rt.distance(p1, p2), sides=16)
        cyl.material = get_material(color)
        cyl.pos = p1
        cyl.dir = rt.normalize(p2 - p1)

    for x1, y1, z1, x2, y2, z2, color in [{lattice_formatted}]:
        p1, p2 = rt.Point3(x1, y1, z1), rt.Point3(x2, y2, z2)
        cyl = rt.Cylinder(radius=0.03, height=rt.distance(p1, p2), sides=8)
        cyl.material = get_material(color)
        cyl.pos = p1
        cyl.dir = rt.normalize(p2 - p1)
    {bz_solid_script}

rt.redrawViews()
print("Export successfully completed!")
'''
        with open(file_path, 'w') as f:
            f.write(script_content)

    @staticmethod
    def export_blender(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data=None):
        atoms_formatted = ",\n        ".join(str(a) for a in atoms_data)
        bonds_formatted = ",\n        ".join(str(b) for b in bonds_data)
        lattice_formatted = ",\n        ".join(str(l) for l in lattice_data)

        bz_solid_script = ""
        if bz_solid_data:
            verts = bz_solid_data['verts']
            faces = bz_solid_data['faces']
            verts_str = ", ".join([f"({p[0]},{p[1]},{p[2]})" for p in verts])
            faces_str = ", ".join([f"({f[0]},{f[1]},{f[2]})" for f in faces])
            bz_solid_script = f'''
bz_mesh_data = bpy.data.meshes.new("BZ_Mesh")
bz_mesh_data.from_pydata([{verts_str}], [], [{faces_str}])
bz_mesh_data.update()
bz_obj = bpy.data.objects.new("BrillouinZone", bz_mesh_data)
crystal_coll.objects.link(bz_obj)

bz_mat = get_material("#FF00FF")
bz_mat.blend_method = 'BLEND'
if bz_mat.node_tree.nodes.get("Principled BSDF"):
    bz_mat.node_tree.nodes.get("Principled BSDF").inputs['Alpha'].default_value = 0.3
bz_obj.data.materials.append(bz_mat)
'''
        script_content = f'''import bpy
import mathutils

coll_name = "TensorSpec_Crystal"
if coll_name not in bpy.data.collections:
    crystal_coll = bpy.data.collections.new(coll_name)
    bpy.context.scene.collection.children.link(crystal_coll)
else:
    crystal_coll = bpy.data.collections[coll_name]

def hex_to_rgba(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)) + (1.0,)

def get_material(hex_color):
    mat_name = "Mat_" + hex_color.replace('#', '')
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = hex_to_rgba(hex_color)
            bsdf.inputs['Roughness'].default_value = 0.15
            bsdf.inputs['Metallic'].default_value = 0.2
    return mat

def create_base_sphere(name, radius, color):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=32, ring_count=16)
    obj = bpy.context.active_object
    obj.data.materials.append(get_material(color))
    bpy.ops.object.shade_smooth()
    mesh = obj.data
    bpy.data.objects.remove(obj)
    return mesh

base_meshes = {{}}
for ax, ay, az, rad, color in [{atoms_formatted}]:
    key = (color, rad)
    if key not in base_meshes: base_meshes[key] = create_base_sphere("AtomMesh", rad, color)
    obj = bpy.data.objects.new("Atom", base_meshes[key])
    obj.location = (ax, ay, az)
    crystal_coll.objects.link(obj)

def create_cylinder_between_points(p1, p2, rad, color, verts):
    vec = p2 - p1
    dist = vec.length
    if dist == 0: return
    bpy.ops.mesh.primitive_cylinder_add(radius=rad, depth=dist, vertices=verts)
    obj = bpy.context.active_object
    obj.data.materials.append(get_material(color))
    bpy.ops.object.shade_smooth()
    obj.location = (p1 + p2) / 2
    obj.rotation_euler = mathutils.Vector((0, 0, 1)).rotation_difference(vec).to_euler()
    bpy.context.collection.objects.unlink(obj)
    crystal_coll.objects.link(obj)

for x1, y1, z1, x2, y2, z2, rad, color in [{bonds_formatted}]:
    create_cylinder_between_points(mathutils.Vector((x1, y1, z1)), mathutils.Vector((x2, y2, z2)), rad, color, 16)

for x1, y1, z1, x2, y2, z2, color in [{lattice_formatted}]:
    create_cylinder_between_points(mathutils.Vector((x1, y1, z1)), mathutils.Vector((x2, y2, z2)), 0.03, color, 8)

{bz_solid_script}
print("Crystal exported successfully!")
'''
        with open(file_path, 'w') as f:
            f.write(script_content)