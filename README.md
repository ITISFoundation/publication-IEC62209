
# 📝 publication-IEC62209

[![CITATION.cff](https://github.com/ITISFoundation/publication-IEC62209/actions/workflows/cff-validator.yml/badge.svg)](https://github.com/ITISFoundation/publication-IEC62209/actions/workflows/cff-validator.yml)
[![Open osparc](https://img.shields.io/badge/²S²PARC-open-blue?style=for-the-badge&logo=Opera)](https://osparc.io)

## A Gaussian-process-model-based approach for robust, independent, and implementation-agnostic validation of complex     multi-variable measurement systems: application to SAR measurement systems
by *Bujard* et al.


  Resource-efficient and robust validation of complex measurement systems that
  would require millions of test permutations for comprehensive coverage is an
  unsolved problem. In the paper, a general, robust, trustworthy, efficient, and
  comprehensive validation approach based on a Gaussian Process model
  (GP) of the test system has been developed that can operate
  system-agnostically, prevents calibration to a fixed set of known validation benchmarks, and
  supports large configuration spaces. The approach includes three steps that can
  be performed independently by different parties: 1) GP model creation, 2) model
  confirmation, and 3) model-based critical search for failures. The new approach
  has been applied to two systems utilizing different measurement methods for
  compliance testing of radiofrequency-emitting devices according to two
  independent standards, i.e., IEC 62209-1528 for
  scanning systems and IEC 62209-3 for array systems.
  The results demonstrate that the proposed measurement system validation is
  practical and feasible. It reduces the effort to a minimum such that it can be
  routinely performed by any test lab or other user and constitutes a pragmatic
  approach for establishing validity and effective equivalence of the two
  measurement device classes.


## Usage

Install python3 dependencies listed in [requirements.txt](requirements.txt)

Run the python3 [executable](bin/iec62209.zip)

