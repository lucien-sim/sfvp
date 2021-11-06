from scipy import integrate, interpolate, optimize
import numpy as np


def shift2d(x: np.ndarray, n: int, axis: int, fill_val: any = 0.) -> np.ndarray:
    """
    Shift array elements n spaces along specified axis. Fill exposed values with fill_val.
    This is basically the equivalent of np.roll, except that values aren't transferred
    from one end of the array to the other.

    :param x: array to be shifted.
    :param n: int, number of spaces to shift array.
    :param axis: int, axis along which array is to be shifted.
    :param fill_val: any, value that will fill exposed elements.
    :return:
    """

    x = np.roll(x, shift=n, axis=axis)
    if n > 0 and axis == 0:
        x[:n, :] = fill_val
    elif n < 0 and axis == 0:
        x[n:, :] = fill_val
    elif n > 0 and axis == 1:
        x[:, :n] = fill_val
    elif n < 0 and axis == 1:
        x[:, n:] = fill_val

    return x


def generate_guess(u: np.ndarray, v: np.ndarray, dx: float, dy: float, method: str = 'randn'):

    if method.lower() == 'randn':

        # Draw values randomly from a normal distribution.
        psi_init = np.random.normal(0, 1, size=(u.shape[0] + 1, u.shape[1] + 1))
        chi_init = np.random.normal(0, 1, size=(u.shape[0] + 1, u.shape[1] + 1))

    elif method.lower() == 'integrate':

        # Guess psi (sf) once, using dpsi = -u_psi * dy, with boundary = dpsi = v_psi * dx
        intx = integrate.cumulative_trapezoid(v, dx=dx, axis=0, initial=0)[0]
        inty = -integrate.cumulative_trapezoid(u, dx=dy, axis=1, initial=0)
        psi1 = intx + inty

        # Guess psi(sf) again, using dpsi = v_psi * dx with boundary = dpsi = -u_psi * dy
        intx = integrate.cumulative_trapezoid(v, dx=dx, axis=0, initial=0)
        inty = -integrate.cumulative_trapezoid(u, dx=dy, axis=1, initial=0)[0:1, 0]
        psi2 = intx + inty

        # Average results and divide by 2
        psi_init = 0.5 * 0.5 * (psi1 + psi2)
        psi_init = psi_init - np.mean(psi_init)

        # Guess chi (vp) once, using dchi = v_chi * dy, with boundary = dchi = u_chi * dx
        intx = integrate.cumulative_trapezoid(u, dx=dx, axis=0, initial=0)[0]
        inty = integrate.cumulative_trapezoid(v, dx=dy, axis=1, initial=0)
        chi1 = intx + inty

        # Guess chi (vp) again, using dchi = u_chi * dx, with boundary = dchi = v_chi * dy
        intx = integrate.cumulative_trapezoid(u, dx=dx, axis=0, initial=0)
        inty = integrate.cumulative_trapezoid(v, dx=dy, axis=1, initial=0)[0:1, 0].ravel()
        chi2 = intx + inty

        # Average the results, then divide by 2 again
        chi_init = 0.5 * 0.5 * (chi1 + chi2)
        chi_init = chi_init - np.mean(chi_init)

        # Pad with zeros so shape is correct
        psi_init = np.pad(psi_init, pad_width=(0, 1), mode='constant')
        chi_init = np.pad(chi_init, pad_width=(0, 1), mode='constant')

    else:
        raise ValueError

    return psi_init, chi_init


def combine_psi_chi(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """Stack psi and chi into a 1d vector."""
    return np.hstack([psi.ravel(), chi.ravel()])


def extract_psi_chi(params: np.ndarray, shape_psi: tuple) -> tuple:
    """Extract psi, chi from 1d vector."""
    psi = params[:np.product(shape_psi)].reshape(shape_psi)
    chi = params[np.product(shape_psi):].reshape(shape_psi)
    return psi, chi


def calculate_velocity(psi: np.ndarray, chi: np.ndarray, dx: float, dy: float) -> tuple:
    """
    Calculate velocity given stream function and velocity potential fields.
    :param psi: array of stream function
    :param chi: array of velocity potential
    :param dx: x grid spacing
    :param dy: y grid spacing
    :return: tuple, (u, v).
    """
    uhat = ((chi[1:, :-1] - chi[:-1, :-1]) / dx + (chi[1:, 1:] - chi[:-1, 1:]) / dx -
            (psi[:-1, 1:] - psi[:-1, :-1]) / dy - (psi[1:, 1:] - psi[1:, :-1]) / dy) * 0.5
    vhat = ((chi[:-1, 1:] - chi[:-1, :-1]) / dy + (chi[1:, 1:] - chi[1:, :-1]) / dy +
            (psi[1:, :-1] - psi[:-1, :-1]) / dx + (psi[1:, 1:] - psi[:-1, 1:]) / dx) * 0.5
    return uhat, vhat


def calculate_irrotational_flow(chi: np.ndarray, dx: float, dy: float) -> tuple:
    """
    Calculate irrotational flow given velocity potential field.
    :param chi: array of velocity potential
    :param dx: float, x grid spacing
    :param dy: float, y grid spacing
    :return: tuple, (irrotational u, irrotational v)
    """

    u_irrot = ((chi[1:, :-1] - chi[:-1, :-1]) / dx + (chi[1:, 1:] - chi[:-1, 1:]) / dx) * 0.5
    v_irrot = ((chi[:-1, 1:] - chi[:-1, :-1]) / dy + (chi[1:, 1:] - chi[1:, :-1]) / dy) * 0.5

    return u_irrot, v_irrot


def calculate_nondivergent_flow(psi: np.ndarray, dx: float, dy: float) -> tuple:
    """
    Calculate nondivergent flow given stream function field.
    :param psi:
    :param dx:
    :param dy:
    :return:
    """
    u_nondiv = -((psi[:-1, 1:] - psi[:-1, :-1]) / dy + (psi[1:, 1:] - psi[1:, :-1]) / dy) * 0.5
    v_nondiv = ((psi[1:, :-1] - psi[:-1, :-1]) / dx + (psi[1:, 1:] - psi[:-1, 1:]) / dx) * 0.5
    return u_nondiv, v_nondiv


def cost_fcn(params: np.ndarray, u: np.ndarray, v: np.ndarray, dx: float, dy: float, lam: float,
            shape_psi: tuple) -> float:
    """
    Calculates cost function.

    :param params: vector of parameters.
    :param u: array of u velocities
    :param v: array of v velocities
    :param dx: grid spacing in x direction
    :param dy: grid spacing in y direction
    :param lam: positive float, regularization parameter
    :param shape_psi: tuple, shape of padded psi array.
    :return: float, value of cost function.
    """

    psi, chi = extract_psi_chi(params, shape_psi)
    uhat, vhat = calculate_velocity(psi, chi, dx, dy)

    return np.sum((uhat - u) ** 2 + (vhat - v) ** 2) + np.sum(lam * (psi ** 2 + chi ** 2))


def jac(params: np.ndarray, u: np.ndarray, v: np.ndarray, dx: float, dy: float, lam: float,
        shape_psi: tuple) -> np.ndarray:
    """
    Calculates partial derivatives of cost function with respect to each parameter.

    :param params: vector of parameters.
    :param u: array of u velocities
    :param v: array of v velocities
    :param dx: grid spacing in x direction
    :param dy: grid spacing in y direction
    :param lam: positive float, regularization parameter
    :param shape_psi: tuple, shape of padded psi array.
    :return: vector of partial derivatives.
    """

    psi, chi = extract_psi_chi(params, shape_psi)
    uhat, vhat = calculate_velocity(psi, chi, dx, dy)

    # Calculate portions that link psi/chi to u/v.
    err_u = np.pad((uhat - u), pad_width=(0, 1), mode='constant')
    err_v = np.pad((vhat - v), pad_width=(0, 1), mode='constant')
    jac_chi = -(err_u - shift2d(err_u, n=1, axis=0)) / dx - shift2d((err_u - shift2d(err_u, n=1, axis=0)) / dx, n=1, axis=1) \
              -(err_v - shift2d(err_v, n=1, axis=1)) / dy - shift2d((err_v - shift2d(err_v, n=1, axis=1)) / dy, n=1, axis=0)
    jac_psi = -(err_v - shift2d(err_v, n=1, axis=0)) / dx - shift2d((err_v - shift2d(err_v, n=1, axis=0)) / dx, n=1, axis=1) \
              +(err_u - shift2d(err_u, n=1, axis=1)) / dy + shift2d((err_u - shift2d(err_u, n=1, axis=1)) / dy, n=1, axis=0)
    jac_psi_chi = combine_psi_chi(jac_psi, jac_chi)

    # Add portion related to magnitude of psi/chi
    jac = jac_psi_chi + 2 * lam * params

    return jac


def sfvp(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray,
         lam: float = 1e-8, guess_method: str = 'randn', interp: bool = False) -> tuple:
    """
    Function for calculating stream function (psi) and velocity potential (chi) fields, given u and v.
    Note that u and v must be on a REGULAR grid (i.e. even grid spacing in x, even grid spacing in y).

    :param u: array of velocity in x component
    :param v: array of velocity in y component
    :param x: x coordinate vector
    :param y: y coordinate vector
    :param lam: float greater than zero, regularization coefficient
    :param guess_method: str, method to generate initial guess. Only 'randn' and 'integrate'
    are implemented. 'randn' assigns values from random normal distribution with mean 0 and std 1.
    'integrate' makes a guess based on the wind field.
    :param interp: bool, default True. Interpolate back to original x and y if true.
    :return: tuple, (psi, chi)
    """

    # Calc dx and dy
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Make initial guess
    psi_guess, chi_guess = generate_guess(u, v, dx, dy, method=guess_method)
    init_params = combine_psi_chi(psi_guess, chi_guess)
    shape_psi = psi_guess.shape

    # Optimize
    opt = optimize.minimize(
        fun=cost_fcn,
        x0=init_params,
        args=(u, v, dx, dy, lam, shape_psi),
        method='Newton-CG',
        jac=jac
    )

    # Get psi and chi.
    psi, chi = extract_psi_chi(opt.x, shape_psi)

    if interp:
        # Interpolate psi and chi to x, y grid.
        x_stag = np.hstack([x[0:1] - 0.5 * dx, x + 0.5 * dx])
        y_stag = np.hstack([y[0:1] - 0.5 * dx, y + 0.5 * dx])
        f = interpolate.interp2d(x=x_stag, y=y_stag, z=psi, kind='linear')
        psi = f(x, y)
        f = interpolate.interp2d(x=x_stag, y=y_stag, z=chi, kind='linear')
        chi = f(x, y)

    return psi, chi


def example():

    x = np.linspace(0, 360, 181)
    y = np.linspace(0, 180, 91)

    X, Y = np.meshgrid(x, y)

    u = (np.sin(X / 20) - np.sin(Y / 20)) * 10
    v = (np.cos(X / 20 + 5) + np.sin(Y / 20 - 5)) * 10

    sf, vp = sfvp(u, v, x, y, lam=1e-8, guess_method='randn', interp=True)
