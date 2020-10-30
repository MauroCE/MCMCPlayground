from subprocess import check_output
import numpy as np


def site_loglike(site, params, isss=5000, is_iter=1, pop="growing"):
    """Computes log likelihood for one site."""
    np.savetxt("paramfile", params, newline=' ', fmt="%.15f")
    target = pop + "/gtree_file" + str(site)
    command = ["./is_moments", target, str(is_iter), str(isss), "paramfile", "1"]
    return np.log(np.float(check_output(command)))


def full_loglike(sites, params, isss=5000, is_iter=1, pop="growing"):
    """Computes full log likelihood."""
    np.savetxt("paramfile", params, newline=' ', fmt="%.15f")
    chunk_files = [pop + "/gtree_file" + str(site) for site in sites]
    ll = 0.0
    for target in chunk_files:
        ll = ll + np.log(np.float(check_output(["./is_moments", target, str(is_iter), str(isss), "paramfile", "1"])))
    return ll
