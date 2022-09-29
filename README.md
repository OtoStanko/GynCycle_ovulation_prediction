# Menstrual Cycle model published by Fischer-Holzhausen et al. 2022 (https://doi.org/10.1016/j.jtbi.2022.111150)

## Description 
Mathematical model of the female menstrual cycle that couples the dynamics between follicles and hormones.

![Alt text](https://github.com/SoFiwork/GynCycle_newVersion/blob/main/Flowchart.jpg)

## Installation
The model is written in MATLAB - no additional package installations are necessary.

To use this GynCycle version, please clone the repository:

```
git clone https://github.com/SoFiwork/GynCycle_newVersion
```

## Usage

This model is an extended version of the GynCycle model by Röblitz et al. 2013 (https://doi.org/10.1016/j.jtbi.2012.11.020). 

MATLAB_model contains all files to run simulations.
To run a simultion use StartSimulation.m. 
Chose one simulation mode: 
-NormalCycle
Stimulation protocols (for details see Fischer et al. 2021)
-LutStim
-FollStim
-DoubStim
To create a populations of models (to run those you need FSHS.txt and StartTimesPoiss.txt with follicle specific parameters):
-Foll_ModelPop
-Horm_ModelPop

CreateFollicles.m create files with follicle-specific parameter (FSHS.txt and StartTimesPoiss.txt).

Algorithm 1 in Fischer-Holzhausen et al. 2022 describes all functionalities called in Simulation.m 

## Authors
Sophie Fischer-Holzhausen
Susanna Röblitz

## License
This model is licensed under the MIT License.

## Citation

@article{fischer2022hormonal,
  title={Hormonal regulation of ovarian follicle growth in humans: Model-based exploration of cycle variability and parameter sensitivities},
  author={Fischer-Holzhausen, Sophie and R{\"o}blitz, Susanna},
  journal={Journal of Theoretical Biology},
  pages={111150},
  year={2022},
  publisher={Elsevier}
}

@article{fischer2021mathematical,
  title={Mathematical modeling and simulation provides evidence for new strategies of ovarian stimulation},
  author={Fischer, Sophie and Ehrig, Rainald and Sch{\"a}fer, Stefan and Tronci, Enrico and Mancini, Toni and Egli, Marcel and Ille, Fabian and Kr{\"u}ger, Tillmann HC and Leeners, Brigitte and R{\"o}blitz, Susanna},
  journal={Frontiers in endocrinology},
  volume={12},
  pages={613048},
  year={2021},
  publisher={Frontiers Media SA}
}
