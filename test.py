import pyvista as pv

print("Starting 3D render test...")

# Create a simple sphere with the glossy 3ds Max settings you want
sphere = pv.Sphere()

# Plot it in a standalone window
plotter = pv.Plotter()
plotter.add_mesh(sphere, color="teal", pbr=True, metallic=0.6, roughness=0.2)
plotter.show()

print("Test complete.")