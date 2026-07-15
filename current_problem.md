[] I open the crystal suite, I am using the PyVista in Windows machine.
    [] 1. the PBR shiny toggle seems not to work
        [x] you gave solution but it gives me this error. (TensorSpec_env) PS C:\Users\sandy\Documents\GitHub\TensorSpec> py -m tensorspec.gui.main_browser
🔌 Crystal Suite: Routing to high-performance PyVista backend.
🖥️ Hardware Detected: Windows (AMD64)
🚀 Engaging High-Performance graphics for Windows.
C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1769: UserWarning: No data to use for scale. scale will be set to False.
  warnings.warn('No data to use for scale. scale will be set to False.')
C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1802: UserWarning: No vector-like data to use for orient. orient will be set to False.
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\gui\suites\crystal_suite.py", line 380, in refresh_render
    self.renderer.draw_atoms(render_struct, self.active_colors, scale_mod=scale, is_shiny=is_shiny, erased_atoms=self.erased_atoms)
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\plotting\backends\pyvista_engine.py", line 109, in draw_atoms
    self.plotter.enable_image_based_lighting() # Prevents black screen
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvistaqt\rwi.py", line 426, in __getattr__
    raise AttributeError(self.__class__.__name__ +
AttributeError: QtInteractor has no attribute named enable_image_based_lighting. Did you mean: 'enable_eye_dome_lighting'?
            [x] you gave solution but it still give error 🚀 Engaging High-Performance graphics for Windows.
C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1769: UserWarning: No data to use for scale. scale will be set to False.
  warnings.warn('No data to use for scale. scale will be set to False.')
C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1802: UserWarning: No vector-like data to use for orient. orient will be set to False.
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\gui\suites\crystal_suite.py", line 380, in refresh_render
    self.renderer.draw_atoms(render_struct, self.active_colors, scale_mod=scale, is_shiny=is_shiny, erased_atoms=self.erased_atoms)
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\plotting\backends\pyvista_engine.py", line 109, in draw_atoms
    self.plotter.enable_image_based_lighting() # Prevents black screen
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvistaqt\rwi.py", line 426, in __getattr__
    raise AttributeError(self.__class__.__name__ +
AttributeError: QtInteractor has no attribute named enable_image_based_lighting. Did you mean: 'enable_eye_dome_lighting'?
            [] the error doesnt come out, it seems like it is trying to do something. but the outcome is still not shiny. maybe it is just the limitation?

    [x] 2. the bond size doesnt work
        [x] you gave solution but it still doesnt work
            [x] you gave solution but it still doesnt work.
                [x] it works now
    
    [x] 3. the threshold doesnt work. in tab_view it is called spin_bond_threshold but in crystal suite it is called bond_threshold
        [x] it works now

[] Tab 4: BZ
    [] seems there are redundants lines drawn while the real BZ is already there. like there is additional diagonal lines or something like that
        [] you gave solution but it gives error like this Traceback (most recent call last):
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\gui\suites\crystal_suite.py", line 690, in handle_draw_bz
    self.renderer.draw_brillouin_zone(scaled_hull_points, np.array(bz_data["simplices"]), style_idx, edges=bz_data["edges"])
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\plotting\backends\pyvista_engine.py", line 272, in draw_brillouin_zone
    line = pv.Line(bz_points[p1_idx], bz_points[p2_idx])
                   ~~~~~~~~~^^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
                [] you gave solution but still error IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
Traceback (most recent call last):
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\gui\suites\crystal_suite.py", line 690, in handle_draw_bz
    self.renderer.draw_brillouin_zone(scaled_hull_points, np.array(bz_data["simplices"]), style_idx, edges=bz_data["edges"])
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\plotting\backends\pyvista_engine.py", line 272, in draw_brillouin_zone
    line = pv.Line(bz_points[p1_idx], bz_points[p2_idx])
                   ~~~~~~~~~^^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
            [] you gave solution but it doesnt work. it gives error (TensorSpec_env) PS C:\Users\sandy\Documents\GitHub\TensorSpec> py -m tensorspec.gui.main_browser
🔌 Crystal Suite: Routing to high-performance PyVista backend.
🖥️ Hardware Detected: Windows (AMD64)
🚀 Engaging High-Performance graphics for Windows.
C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1769: UserWarning: No data to use for scale. scale will be set to False.
  warnings.warn('No data to use for scale. scale will be set to False.')
C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1802: UserWarning: No vector-like data to use for orient. orient will be set to False.
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\gui\suites\crystal_suite.py", line 690, in handle_draw_bz
    self.renderer.draw_brillouin_zone(scaled_hull_points, np.array(bz_data["simplices"]), style_idx, edges=bz_data["edges"])
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\plotting\backends\pyvista_engine.py", line 270, in draw_brillouin_zone
    line = pv.Line(bz_points[int(p1_idx)], bz_points[int(p2_idx)])
                             ^^^^^^^^^^^
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'
            [] still same error C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1769: UserWarning: No data to use for scale. scale will be set to False.
  warnings.warn('No data to use for scale. scale will be set to False.')
C:\Users\sandy\Documents\GitHub\TensorSpec\TensorSpec_env\Lib\site-packages\pyvista\core\filters\data_set.py:1802: UserWarning: No vector-like data to use for orient. orient will be set to False.
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\gui\suites\crystal_suite.py", line 690, in handle_draw_bz
    self.renderer.draw_brillouin_zone(scaled_hull_points, np.array(bz_data["simplices"]), style_idx, edges=bz_data["edges"])
  File "C:\Users\sandy\Documents\GitHub\TensorSpec\tensorspec\plotting\backends\pyvista_engine.py", line 270, in draw_brillouin_zone
    line = pv.Line(bz_points[int(p1_idx)], bz_points[int(p2_idx)])
                             ^^^^^^^^^^^
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'


[x] I tried to load the WTe2 cif file, load it in the dft, do tight binding band structure dispersion, it works. but when I change to 2D kx ky iso energy calculation, it freezes. I think this is due to the memory issue because it is heacy. but I want to know if it is really the case. how to debug it? after waiting sometimes there is a dialogue saying cannot access local variable 'template_name' where it is not associated with a value
    [x] you gave solution and it works
[] now I found that closing arpes suite also give this error like crystal suite before 2026-07-15 07:01:23.150 (1108.455s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: Σ₧áα▓ê╔è
2026-07-15 07:01:23.150 (1108.456s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: Σ║áα▓ê╔è
2026-07-15 07:01:23.150 (1108.456s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: Σ¼áα▓ê╔è
2026-07-15 07:01:23.151 (1108.456s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: σàáα▓ê╔è
2026-07-15 07:01:23.151 (1108.456s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: Σ¼áα▓ê╔è
2026-07-15 07:01:23.151 (1108.456s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: σàáα▓ê╔è
2026-07-15 07:01:23.151 (1108.456s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: σàáα▓ê╔è
2026-07-15 07:01:23.151 (1108.457s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in MakeCurrent(), error: Σòáα▓ê╔è
2026-07-15 07:01:23.152 (1108.457s) [65FB9151CD2EFFCF]vtkWin32OpenGLRenderWin:91     ERR| vtkWin32OpenGLRenderWindow (0000024A0D2005A0): wglMakeCurrent failed in Clean(), error: 6
we should do what we did in crystal suite also to arpes. and also to other app that use similar feature

[] there is stille rror like this. I dont know which one causing it ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 廀Ǭ
2026-07-15 07:49:14.508 (2332.982s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σ╢ÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 嶀Ǭ
2026-07-15 07:49:14.509 (2332.983s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σêÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 刀Ǭ
2026-07-15 07:49:14.510 (2332.984s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σúÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 壀Ǭ
2026-07-15 07:49:14.511 (2332.985s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σ▓ÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 岀Ǭ
2026-07-15 07:49:14.511 (2332.985s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σ╝ÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 开Ǭ
2026-07-15 07:49:14.512 (2332.986s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σúÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 壀Ǭ
close suite Crystal Viewer
2026-07-15 07:49:14.695 (2333.169s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σäÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 儀Ǭ
2026-07-15 07:49:14.705 (2333.179s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σáÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 堀Ǭ
2026-07-15 07:49:14.707 (2333.181s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σçÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 净Ǭ
2026-07-15 07:49:14.709 (2333.183s) [6F95B350B09F4D11]vtkWin32OpenGLRenderWin:256    ERR| vtkWin32OpenGLRenderWindow (000001ECB3326AC0): wglMakeCurrent failed in MakeCurrent(), error: σäÇεêÜ╟¼
ERROR:root:wglMakeCurrent failed in MakeCurrent(), error: 儀Ǭ