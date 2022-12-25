# @Author: Marc Cheneau <marc>
# @Date:   21-07-2020
# @Email:  marc.cheneau@institutoptique.fr
# @Filename: lattice.py
# @Last modified by:   marc
# @Last modified time: 16-12-2020


from math import pi
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import trapz
from scipy.constants import hbar
from multiprocess import Pool

""" Set of tools to solve the 1D lattice problem.

Except otherwise stated, the units used in this module are:
* π/a for quasimomentum coordinates;
* a for real-space coordinates;
* Er = ℏ²π² / 2ma² for energies.

"""


def sites(size):
    """Return the lattice sites' indices.

    Parameters
    ----------
    size : int
        Number of lattice sites.

    Returns
    -------
    : (size,) ndarray
        Lattice sites' indices.
        If size is even they range from -size/2 to size/2-1.
        If size is odd they range from -(size-1)/2 to (size-1)/2.

    """
    return np.arange(-size // 2, size // 2) + size % 2


def quasimomenta(size, include="end"):
    """Return the quasimomenta.

    Parameters
    ----------
    size : int
        Number of lattice sites.
    include : ("start", "end")
        Choose which of q=-1 or q=+1 is included in the output array.

    Returns
    -------
    : (size,) ndarray
        Quasimomenta ranging from -1 to 1.

    """
    assert include in ("start", "end")
    if include == "start":
        return np.linspace(-1, 1, size, endpoint=False)
    else:
        return np.flip(np.linspace(1, -1, size, endpoint=False))


def hamiltonian(V, q, size, full_matrix=False):
    """Return the Hamiltonian in the plane-wave basis.

    Parameters
    ----------
    V : float
        Lattice depth in units of the recoil energy.
    q : float
        Quasi-momentum in units of pi/(lattice period).
    size : int
        Number of lattice sites.
    full_matrix : bool, optional
        If False, diagonal and off-diagonal elements are returned as 1d-arrays.
        If True, the Hamiltian is returned as a 2d-array.

    Returns
    -------
    diag : (size,) array
        Diagonal elements of the Hamiltonian.
        The values are in units of the recoil energy.
    offdiag : (size-1,) array
        Off-diagonal elements of the Hamiltonian.

    If full_matrix is True, the Hamiltonian is returned instead as a (size, size) array.

    """
    diag = V / 2 + (q + 2 * sites(size)) ** 2
    offdiag = -V / 4 * np.ones(size - 1)
    if full_matrix:
        return np.diag(diag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
    else:
        return diag, offdiag


def eigenproblem(
    V, q, size=None, eigvals_only=False, kohns_phase=False, bands=(), squeeze=False
):
    """Retrun the eigenvalues and eigenvectors of the lattice Hamiltonian.

    Parameters
    ----------
    V : float
        Lattice depth.
    q : sequence of floats
        Quasimomenta.
    size : int, optional
        Number of lattice sites. If q is an array then size is set to q.size.
    eigvals_only : bool, optional
        If True, only the eigenenergies are computed.
    kohns_phase : bool, optional
        If True, the phase of the eigenvectors will be changed to obtain real-valued
        Wannier functions.
    bands : int or sequence of ints, optional
        Band indices. If empty (default) all bands are returned.
    squeeze : bool, optional
        Squeeze the output. Default is False.

    Returns
    -------
    w : (Q, B) ndarray
        Eigenvalues, in ascending order and in units of the recoil energy.
        axis 0 spans the quasimomentum.
        axis 1 spans the band index.
    v : (Q, I, B) ndarray
        Normalized eigenvectors (Bloch states).
        axis 0 spans the quasimomentum.
        axis 1 spans the diffraction order.
        axis 2 spans the band index.

    """
    q = np.atleast_1d(q)
    bands = np.atleast_1d(bands)
    if bands.size == 0:
        bands = np.arange(size)

    if q.size == 1:  # scalar
        res = eigh_tridiagonal(*hamiltonian(V, q, size), eigvals_only=eigvals_only)
        if eigvals_only:
            w = res[..., bands]
            return w
        else:
            w = np.expand_dims(res[0], 0)[..., bands]
            v = np.expand_dims(res[1], 0)[..., bands]
    else:
        size = q.size
        assert size is not None, "`size` must have a finite value if `q` is a scalar"

        def _wrapper(q):
            return eigh_tridiagonal(*hamiltonian(V, q, size), eigvals_only=eigvals_only)

        with Pool() as pool:
            res = pool.map(_wrapper, q)
        if eigvals_only:
            w = np.asarray(res)[..., bands]
            return w
        else:
            w = np.asarray([r[0] for r in res])[..., bands]
            v = np.asarray([r[1] for r in res])[..., bands]

    if kohns_phase:
        # Kohn's phase choice for real valued Wannier functions:
        #  even band: Bloch wave @ x=0 real and positive
        #  odd band: first derivative of Bloch wave @ x=0 real and positive
        v = v.astype(complex)
        q = np.expand_dims(q, (1, 2))
        i = np.expand_dims(sites(size), (0, 2))
        n = np.expand_dims(bands, (0, 1))
        sign = np.sign(np.sum(v * np.where(n % 2, q + 2 * i, 1), axis=1, keepdims=True))
        v *= np.where(n % 2, -1j, 1) * sign

    if squeeze:
        return np.squeeze(w), np.squeeze(v)
    else:
        return w, v


def energybands(V, q, size=None, bands=(), squeeze=False):
    """Return the energy bands.

    This function is just a wrapper for `eigenproblem` with `eigvals_only=True`.

    Parameters
    ----------
    V : float
        Lattice depth.
    q : sequence of floats
        Quasimomenta.
    size : int, optional
        Number of lattice sites. If q is an array then size is set to q.size.
    bands : int or sequence of ints, optional
        Band indices. If empty (default) all bands are returned.
    squeeze : bool, optional
        Squeeze the output. Default is False.

    Returns
    -------
    w : (Q, B) ndarray
        Eigenvalues, in ascending order and in units of the recoil energy.
        axis 0 spans the quasimomentum.
        axis 1 spans the band index.

    """
    w = eigenproblem(V, q, size=size, eigvals_only=True, bands=bands, squeeze=squeeze)
    return np.squeeze(w) if squeeze else w


def bloch(x, v, squeeze=True):
    """Return the Bloch wavefunctions.

    Parameters
    ----------
    x : sequence of floats
        Real-space positions.
    v : (Q, I, B) ndarray
        Normalized Bloch states.
        axis 0 spans the quasimomentum.
        axis 1 spans the diffraction order.
        axis 2 spans the band index.
    squeeze : bool, optional
        Squeeze the output.

    Returns
    -------
    bloch : (Q, X, B) ndarray
        Bloch waves.
        axis 0 spans the quasimomentum.
        axis 1 spans the position.
        axis 2 spans the band index.

    """
    size = v.shape[0]
    x = np.expand_dims(np.atleast_1d(x), (0, 2, 3))
    i = np.expand_dims(sites(size), (0, 1, 3))
    q = np.expand_dims(quasimomenta(size), (1, 2, 3))
    v = np.expand_dims(v, 1)
    bloch = np.sum(np.exp(1j * pi * (q + 2 * i) * x) * v, axis=2)

    return np.squeeze(bloch) if squeeze else bloch


def wannier(x, v, squeeze=True):
    """Return the Wannier wavefunctions.

    Parameters
    ----------
    x : sequence of floats
        Real-space positions.
    v : (Q, I, B) ndarray
        Normalized Bloch states.
        axis 0 spans the quasimomentum.
        axis 1 spans the diffraction order.
        axis 2 spans the band index.
    squeeze : bool, optional
        Squeeze the output.

    Returns
    -------
    wannier : (X, B) ndarray
        Wannier function.
        axis 0 spans the position.
        axis 1 spans the band index.

    """
    wannier = np.sum(bloch(x, v, squeeze), axis=0) / v.shape[0]
    return np.real_if_close(wannier)


def hubbardU(x, v, scale=None, squeeze=True):
    """Form factor for the onsite energy in the Hubbard model in a 1D lattice.

    The form factor is the integral ∫ w(x)**4 dx.

    Parameters
    ----------
    x : sequence of floats
        Real-space positions over which the integration is performed.
    v : (Q, I, B) ndarray
        Normalized Bloch states.
        axis 0 spans the quasimomentum.
        axis 1 spans the diffraction order.
        axis 2 spans the band index.
    scale : float or None, optional
        Length scale for proper normalization.
    squeeze : bool, optional
        Squeeze the output.

    Returns
    -------
    U : float or (B,) ndarray
        Form factor for the onsite energy.

    """
    w = wannier(x, v, squeeze)
    U = trapz(np.abs(w) ** 4, x=x, axis=0)
    if scale is not None:
        U /= scale

    return np.squeeze(U) if squeeze else U


def hubbardJ(w, d=1, squeeze=True):
    """Tunnel coupling between two sites distant by d in the Hubbard model.

    The coupling J(d) is obtained by projecting the band energy E(q) on the function
    cos(πdq) thanks to the relation: E(q) = 2 ∑_d J(d) cos(πdq).

    Parameters
    ----------
    w : (Q, B) ndarray
        Bloch state energies.
        axis 0 spans the quasimomentum.
        axis 1 spans the band index.
    d : int
        Distance between the coupled sites. d=1 means nearest neighbours.
    squeeze : bool, optional
        Squeeze the output.

    Returns
    -------
    J : float or (B,) ndarray
        Tunnel coupling.

    """
    size = w.shape[0]
    assert 0 < d, "d must be > 0"
    assert d < size, f"d exceeds the lattice size ({size - 1})"

    q = quasimomenta(size)
    q = np.expand_dims(q, [a + q.ndim for a in range(w.ndim - q.ndim)])
    J = np.abs(np.sum(np.cos(d * np.pi * q) * w, axis=0)) / d / q.size

    return np.squeeze(J) if squeeze else J


def recoilenergy(a, m):
    """Return the recoil energy of the lattice.

    The recoil energy is defined as: Er = ℏ²π²/2ma².

    Parameters
    ----------
    a : float or ndarray
        Lattice period in [m].
    m : float or ndarray
        Atomic mass in [kg].

    Returns
    -------
    : float or ndarray
        Recoil energy in [J].

    """

    return hbar ** 2 * (pi / a) ** 2 / 2 / m


def sitefreq(V, a, m):
    """Return the frequency associated with the harmonic motion in one lattice site.

    The frequency ν=ω/2π is defined by the equality: mω²x²/2 ≃ V sin²(πx/a).

    Parameters
    ----------
    V : float or ndarray
        Lattice depth in [J].
    a : float or ndarray
        Lattice period in [m].
    m : float or ndarray
        Atomic mass in [kg].

    Returns
    -------
    : float or ndarray
        Frequency of the harmonic motion in [Hz].

    """

    return (hbar * (pi / a) ** 2 / m) * np.sqrt(V / recoilenergy(a, m)) / (2 * np.pi)
