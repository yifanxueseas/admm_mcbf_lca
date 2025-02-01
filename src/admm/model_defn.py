import jax.numpy as jnp
import numpy as np


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def bicycle(x, u, t):
    """
    We assume system params L = 1m and link to model: https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html
    :param x: (x, y, theta, v)
    :param u: (delta, a)
    :param t: time step
    :return: dynamics
    """
    px, py, theta, v = x
    theta = angle_wrap(theta)
    delta, acc = u
    return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), v * jnp.tan(delta), acc])


def modified_unicycle(x, u, t):
    """
    Using the modified unicycle model with a = 0.3
    :param x: (x, y, theta)
    :param u: (v, omega)
    :param t: time step
    :return: dynamics
    """
    a = 0.3
    px, py, theta = x
    theta = angle_wrap(theta)
    v, omega = u
    return jnp.array([v * jnp.cos(theta) - a * omega * jnp.sin(theta), v * jnp.sin(theta) + a * omega * jnp.cos(theta), omega])


def unicycle(x, u, t):
    """
    :param x: (x, y, theta)
    :param u: (v, omega)
    :param t: time step
    :return: dynamics
    """
    px, py, theta = x
    v, omega = u
    theta = angle_wrap(theta)
    return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega])


def dubins(x, u, t):
    px, py, v, psi, w = x
    # px, py, v, w, psi = x
    psi = angle_wrap(psi)
    ul, ua = u
    return jnp.array(
        [v * jnp.cos(psi) - 0.01 * w * jnp.sin(psi), v * jnp.sin(psi) + 0.01 * w * jnp.cos(psi), ul - 0.01 * w ** 2, w,
         ua])