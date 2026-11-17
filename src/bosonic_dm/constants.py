# Copyright (C) 2026 Francesco Borra
#

"""Physical constants and detector naming shared across bosonic-DM."""

from __future__ import annotations

import math

# Electron mass in keV.
M_E_KEV = 510.99895

# Q-value of the Ge-76 double-beta decay in keV.
Q_GE76_KEV = 2039.0

# Endpoint of the Ar-39 beta spectrum in keV.
AR39_ENDPOINT_KEV = 565.8

# Conversion factor from Gaussian FWHM to standard deviation.
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

# Detector type by the first letter of the detector name.
DET_TYPE_MAP = {"B": "BEGe", "C": "COAX", "V": "ICPC", "P": "PPC"}
