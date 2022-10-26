.. _faq:

Implementation details, conventions, and FAQ
============================================

Sign of the mixer Hamiltonian
    In the original paper on QAOA (`Ref 1 <#references>`__), Farhi `et al` use :math:`\sum_i \hat{X}_i` as
    the mixer Hamiltonian, with the initial state being its maximum eigenstate :math:`\left|+ \cdots +\right>`. 
    In OpenQAOA, we instead choose our mixer Hamiltonian to be :math:`-\sum_i \hat{X}_i`, so that the initial state 
    :math:`\left|+ \cdots +\right>` is now its minimum energy eigenstate. Conceptually this makes the analogy to adiabatic
    computing clear, since we seek to transform from the ground state of the mixer Hamiltonian to the ground state of the cost Hamiltonian. 

Implementation of circuit rotation angles
    In quantum mechanics, the basic time evolution operator is :math:`\exp(-iHt)` for a Hamiltonian `H` and total
    evolution time `t`. Generically, in the QAOA mixer Hamiltonian, the operator :math:`-X` is to be applied for a total time 
    :math:`\beta`, which is one of the parameters we seek to optimise. We therefore need to implement the time evolution 
    :math:`\exp(i\beta X)`, which can be achieved using the RX(:math:`\theta`) operator if we set :math:`\theta = -2\beta`. 

    Similarly, the cost Hamiltonian operator :math:`\exp(-i\gamma hZ)` can be implemented via an RZ(:math:`\theta`) rotation, setting
    :math:`\theta = 2\gamma h`. You can verify these details in the methods that relate to the creation of these angles in ``openqaoa.qaoa_parameters.standardparams.QAOAVariationalStandardParams``.

Where does the factor ``0.7 * n_steps`` in the ``linear_ramp_from_hamiltonian()`` method come from?
    The ``.linear_ramp_from_hamiltonian()`` parameters are inspired by analogy between
    QAOA and a discretised adiabatic annealing process. If we pick a linear ramp annealing schedule, i.e. :math:`s(t) = \frac{t}{\tau}`, where :math:`\tau` is the total
    annealing time, we need to specify two numbers: the total annealing time :math:`\tau` and the step width
    :math:`\Delta t`. Equivalently, we can also specify the total annealing time :math:`\tau` together with
    the number of steps :math:`n_{\textrm{steps}}`, which is also called `p` in the
    context of QAOA. A good discretised annealing schedule has to strike a
    balance between a long annealing time :math:`\tau` and a small step width
    :math:`\Delta t = \frac{\tau}{n_{\textrm{steps}}}`. We have found in numerical
    experiments that :math:`\Delta t = 0.7 = \frac{\tau}{n_{\textrm{steps}}}` strikes a reasonably good balance
    for many problem classes and instances, at least for the small system sizes one can feasibly simulate.
    For larger systems or smaller energy gaps, it might be necessary to choose smaller values of :math:`\Delta t`.
    The implementation of this method can be found in the subclasses of ``QAOAVariationalBaseParams``.

Discrete sine and cosine transformations for the ``QAOAVariationalFourierParams`` class
    In converting between the :math:`\beta` and :math:`\gamma` parameters of the ``QAOAVariationalStandardParams`` class, and the `u` and `v` parameters of the 
    ``QAOAVariationalFourierParams`` class, we use the type II versions of the discrete sine and cosine transformations. These are included in Scipy's fast Fourier 
    transforms module `fftpack <https://docs.scipy.org/doc/scipy-0.14.0/reference/fftpack.html>`_. With the conventions used therein, in OpenQAOA the transformations are then given by:

    .. math::

	\gamma_i = 2 \sum_{k=0}^{q-1} u_k
		      \sin\left[
		             (k + 1/2)
    			     (i+1)			
                             \frac{\pi}{p}
		          \right]

	\beta_i = 2 \sum_{k=0}^{q-1} v_k
		      \cos\left[
		            (2k + 1) 
		            i\frac{\pi}{2p}
		          \right]
 
    While these differ from the versions used in `Ref 2 <#references>`__, this is merely a convention.


References
----------

1. E. Farhi et al, `A Quantum Approximate Optimization Algorithm <https://arxiv.org/abs/1411.4028>`__
2. L. Zhou et al, `Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices <https://arxiv.org/abs/1812.01041>`__ 