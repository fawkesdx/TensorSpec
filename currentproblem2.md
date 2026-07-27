These features are what I want. they are not related to each other

1. I want to have option where the orbital of the atoms are drawn on each of the atom in the crystal_suite.py . the orbital comes from the DFT calculation (if it is true) of the dft_suite. I will give you the crystal_suite.py and the tab py


2. I want to have the option to define at which surface direction is my arpes is performed in the arpes_suite.py . so I am thinking to allow the cif file of bulk be calculated in the DFT calculation but once it reach the arpes suite, I want the arpes suite to define the surface direction. and then the surface direciton will define the kx and the ky direction according to the surface axis. this option should be input-able as hkl as the hkl is defined in the crystal suite. so the arpes simulation is performed with respect to the kx ky of this cleaved surface directions. I will give you the arpes_suite and th e arpes_panel


if you need more files to be uploaded, tell me what file