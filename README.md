# Magnetic Anisotropy Energy Densities

Calculates magnetic anisotropy density of collinear systems

If you use this code, please cite: https://arxiv.org/abs/2205.00300.
In that work, several examples are given for FeCl2, VSe2 and CrI3. 

The code here post-processes Quantum Espresso (QE) output and works with versions > 7.0.0.
Required QE output are data-file-schema.xml and atomic-proj.xml files that are 
produced at the end of projwfc.x runs. 

Here is a typical example of how the code can be used:


```python
#!/usr/bin/env python3
import sys
mae_path = '<MAE.py directory>'
sys.path.insert(0, mae_path)
from mae import MAE
mae = MAE(filename_data = './data-file-schema.xml',
          filename_proj = './atomic_proj.xml',
          wfc_dict = {'Cr':['s','s','p','d'], 
                     'I':['s','p']})
proj = mae.get_proj(['Cr1_d'])
res = mae.get_mat(proj)
mae.plot_dos_1D(res['Cr1_d'])
mae.plot_dos_2D(res['Cr1_d'])
```
This block of code above will create two plots for the MAE density
around the d-orbitals of first Cr atom in CrI3. 

wfc_dict is the dictionary of atomic wavefunctions (in the same order)
for each pseudopotential used in the calculation.

1D plot is the marginal MAE densities on the valence and conduction bands.
2D plot is pair MAE density.

1D plot is similar to the Figures 4c, 4g, 6c and 6d in https://arxiv.org/abs/2205.00300.
2D plot is similar to the Figure 5 in the same manuscript. 
