# Guided Artery Segmentation on 3D Slicer

## Prereq
- Slicer 5.6.1
- Extension
    - https://github.com/lassoan/SlicerSegmentEditorExtraEffects
    - https://github.com/vmtk/SlicerExtension-VMTK
    - https://github.com/Slicer/SlicerJupyter

## Setup
1. `git clone git@github.com:flapperz/artery-seg.git`
2. Download files from [Chula OneDrive](https://chula-my.sharepoint.com/:f:/g/personal/6472006221_student_chula_ac_th/EjkmwTnvnRxFiJ_U-Z75CDYBS2YUwHnmELWbFL0ugazV4A?e=fBUXAp) and put in `./data`
3. Setup SlicerJupyter 
    
    ```bash
     jupyter-kernelspec install "/Applications/Slicer4.app/Contents/Extensions-29738/SlicerJupyter/share/Slicer-4.11/qt-loadable-modules/JupyterKernel/Slicer-4.11" --replace --user
    ```