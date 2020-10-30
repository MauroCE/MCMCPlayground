from SimulationCode.likelihood_functions import site_loglike
import numpy as np

params = np.array([4.605170000000000, 2.995732000000000, -2.302585000000000])
print(site_loglike(1, params))
