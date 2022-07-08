from __future__ import annotations

import numpy as np

from copy import deepcopy
from .basebackend import QAOABaseBackendShotBased

def log_qfim_evals(logger: Logger) -> Logger:
    
    current_total_eval = logger.func_evals.best[0]
    current_total_eval += 1
    current_qfim_eval = logger.qfim_func_evals.best[0]
    current_qfim_eval += 1
    logger.log_variables({'func_evals': current_total_eval, 
                          'qfim_func_evals': current_qfim_eval})
    
    return logger

def qfim(backend_obj: QAOABaseBackend, 
         params: QAOAVariationalBaseParams, 
         logger: Logger, 
         eta: float = 0.00000001):
    
        """
        Returns a callable qfim_fun(args) that computes the quantum fisher information matrix at `args` according to :
        $$[QFI]_{ij} = Re(<∂iφ|∂jφ>) − <∂iφ|φ><φ|∂jφ>$$.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of
            the parameter classes, containing hyperparameters and variable parameters).

        eta: `float`
            The infinitesimal shift used to compute `|∂jφ>`, the partial derivative 
            of the wavefunction w.r.t a parameter. 

        Returns
        -------
        qfim_array:
            The quantum fisher information matrix, a 2p*2p symmetric square matrix 
            with elements [QFI]_ij = Re(<∂iφ|∂jφ>) − <∂iφ|φ><φ|∂jφ>.
        """
        
        if isinstance(backend_obj, QAOABaseBackendShotBased):
            raise NotImplementedError("QFIM computation is not currently available on shot-based")
        
        psi = backend_obj.wavefunction(params)
        log_qfim_evals(logger)
        
        qfim_array = np.zeros((len(params), len(params)))

        copied_params = deepcopy(params)

        def qfim_fun(args):

            for i in range(len(args)):
                for j in range(i+1):
                    vi, vj = np.zeros(len(args)), np.zeros(len(args))
                    vi[i] = eta
                    vj[j] = eta

                    copied_params.update_from_raw(args + vi)
                    wavefunction_plus_i = np.array(
                        backend_obj.wavefunction(copied_params))
                    log_qfim_evals(logger)

                    copied_params.update_from_raw(args - vi)
                    wavefunction_minus_i = np.array(
                        backend_obj.wavefunction(copied_params))
                    log_qfim_evals(logger)

                    copied_params.update_from_raw(args + vj)
                    wavefunction_plus_j = np.array(
                        backend_obj.wavefunction(copied_params))
                    log_qfim_evals(logger)

                    copied_params.update_from_raw(args - vj)
                    wavefunction_minus_j = np.array(
                        backend_obj.wavefunction(copied_params))
                    log_qfim_evals(logger)

                    di_psi = (wavefunction_plus_i - wavefunction_minus_i)/eta
                    dj_psi = (wavefunction_plus_j - wavefunction_minus_j)/eta

                    qfim_array[i][j] = np.real(
                        np.vdot(di_psi, dj_psi)) - np.vdot(di_psi, psi)*np.vdot(psi, dj_psi)

                    if i != j:
                        qfim_array[j][i] = qfim_array[i][j]

            return qfim_array

        return qfim_fun