import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 15
plt.style.use("dark_background")


# Question 1
def f(x1, x2):
    return 12.069 * x1**2 + 21.504 * x2**2 - 1.7321 * x1 - x2


def dfdx1_fd(x1, x2, eps=1e-3):
    """Compute the derivative of f with respect to x1 using forward differences."""
    f_plus_val = f(x1 + eps, x2)
    f_val = f(x1, x2)
    return (f_plus_val - f_val) / eps


def dfdx2_fd(x1, x2, eps=1e-3):
    """Compute the derivative of f with respect to x2 using forward differences."""
    f_plus_val = f(x1, x2 + eps)
    f_val = f(x1, x2)
    return (f_plus_val - f_val) / eps


def dfdx1_bd(x1, x2, eps=1e-3):
    """Compute the derivative of f with respect to x1 using backward differences."""
    f_val = f(x1, x2)
    f_minus_val = f(x1 - eps, x2)
    return (f_val - f_minus_val) / eps


def dfdx2_bd(x1, x2, eps=1e-3):
    """Compute the derivative of f with respect to x2 using backward differences."""
    f_val = f(x1, x2)
    f_minus_val = f(x1, x2 - eps)
    return (f_val - f_minus_val) / eps


def dfdx1_cd(x1, x2, eps=1e-3):
    """Compute the derivative of f with respect to x1 using central differences."""
    f_plus_val = f(x1 + eps, x2)
    f_minus_val = f(x1 - eps, x2)
    return (f_plus_val - f_minus_val) / (2 * eps)


def dfdx2_cd(x1, x2, eps=1e-3):
    """Compute the derivative of f with respect to x2 using central differences."""
    f_plus_val = f(x1, x2 + eps)
    f_minus_val = f(x1, x2 - eps)
    return (f_plus_val - f_minus_val) / (2 * eps)


def grad(x1, x2, eps=1e-3, method="cd"):
    """
    Calculate the gradient of f at (x1, x2) using the specified method.

    Parameters
    ----------
    x1 : float
        The value of x1 at which to evaluate the gradient.
    x2 : float
        The value of x2 at which to evaluate the gradient.
    eps : float, optional
        The step size to use for the finite difference method.
    method : str, optional
        The method to use for calculating the gradient. Must be one of
        "fd" (forward differences), "bd" (backward differences), or
        "cd" (central differences).

    Returns
    -------
    grad : ndarray
        The gradient of f at (x1, x2).
    """
    if method == "fd":
        return np.array([dfdx1_fd(x1, x2, eps), dfdx2_fd(x1, x2, eps)])
    elif method == "bd":
        return np.array([dfdx1_bd(x1, x2, eps), dfdx2_bd(x1, x2, eps)])
    elif method == "cd":
        return np.array([dfdx1_cd(x1, x2, eps), dfdx2_cd(x1, x2, eps)])
    else:
        raise ValueError("Unknown method")


x1 = 5
x2 = 6
eps = 1e-2
analytic_grad = np.array([24.138 * x1 - 1.7321, 43.008 * x2 - 1])
grad_fd = grad(x1, x2, method="fd", eps=eps)
grad_bd = grad(x1, x2, method="bd", eps=eps)
grad_cd = grad(x1, x2, method="cd", eps=eps)
print(f"Results for eps = {eps}")
print(f"Analytic gradient of f at x1 = {x1}, x2 = {x2}: {analytic_grad}")
print("----------------------" * 3)
print(f"Gradient of f at x1 = {x1}, x2 = {x2} using forward differences: {grad_fd}")
print(f"Gradient of f at x1 = {x1}, x2 = {x2} using backward differences: {grad_bd}")
print(f"Gradient of f at x1 = {x1}, x2 = {x2} using central differences: {grad_cd}")

fd_error = np.linalg.norm(analytic_grad - grad_fd)
bd_error = np.linalg.norm(analytic_grad - grad_bd)
cd_error = np.linalg.norm(analytic_grad - grad_cd)

print("Errors")
print("----------------------" * 3)
print(f"Error in gradient using forward differences: {fd_error}")
print(f"Error in gradient using backward differences: {bd_error}")
print(f"Error in gradient using central differences: {cd_error}")

epsilons = np.logspace(-8, 0, 200)
fd_errors = np.zeros(epsilons.shape)
bd_errors = np.zeros(epsilons.shape)
cd_errors = np.zeros(epsilons.shape)

for i, eps in enumerate(epsilons):
    fd_errors[i] = np.linalg.norm(analytic_grad - grad(x1, x2, method="fd", eps=eps))
    bd_errors[i] = np.linalg.norm(analytic_grad - grad(x1, x2, method="bd", eps=eps))
    cd_errors[i] = np.linalg.norm(analytic_grad - grad(x1, x2, method="cd", eps=eps))

plt.figure(figsize=(10, 8))
plt.loglog(epsilons, fd_errors, label="Forward difference", color="red")
plt.loglog(epsilons, bd_errors, label="Backward difference", color="green")
plt.loglog(epsilons, cd_errors, label="Central difference", color="yellow")
plt.xlabel("$\epsilon$ (Log Scale)")
plt.ylabel("Error (Log Scale)")
plt.title("Error in gradient of $f(x_1, x_2)$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("plots/gradient_error_p1.png")


# Question 2


def f(x1, x2):
    numerator = 4 * x2**2 - x1 * x2
    denominator = 1000 * (x2 * x1**3 - x1**4)
    return numerator / (denominator + 1e-15)


def gradf(x1, x2):
    t11 = -(3 * x2 * (x1**2 - 6 * x2 * x1 + 4 * x2**2))
    t12 = 1000 * x1**4 * (x1 - x2) ** 2
    t1 = t11 / (t12 + 1e-15)

    t21 = 4 * x2**2 - 8 * x1 * x2 + x1**2
    t22 = 1000 * x1**3 * (x2 - x1) ** 2
    t2 = t21 / (t22 + 1e-15)

    return np.array([t1, t2])


x1 = 0.5
x2 = 1.5
eps = 1e-2
analytic_grad = gradf(x1, x2)
grad_fd = grad(x1, x2, method="fd", eps=eps)
grad_bd = grad(x1, x2, method="bd", eps=eps)
grad_cd = grad(x1, x2, method="cd", eps=eps)
print(f"Results for eps = {eps}")
print(f"Analytic gradient of f at x1 = {x1}, x2 = {x2}: {analytic_grad}")
print("----------------------" * 3)
print(f"Gradient of f at x1 = {x1}, x2 = {x2} using forward differences: {grad_fd}")
print(f"Gradient of f at x1 = {x1}, x2 = {x2} using backward differences: {grad_bd}")
print(f"Gradient of f at x1 = {x1}, x2 = {x2} using central differences: {grad_cd}")

fd_error = np.linalg.norm(analytic_grad - grad_fd)
bd_error = np.linalg.norm(analytic_grad - grad_bd)
cd_error = np.linalg.norm(analytic_grad - grad_cd)

print("Errors")
print("----------------------" * 3)
print(f"Error in gradient using forward differences: {fd_error}")
print(f"Error in gradient using backward differences: {bd_error}")
print(f"Error in gradient using central differences: {cd_error}")

epsilons = np.logspace(-8, -1, 200)
fd_errors = np.zeros(epsilons.shape)
bd_errors = np.zeros(epsilons.shape)
cd_errors = np.zeros(epsilons.shape)

for i, eps in enumerate(epsilons):
    fd_errors[i] = np.linalg.norm(analytic_grad - grad(x1, x2, method="fd", eps=eps))
    bd_errors[i] = np.linalg.norm(analytic_grad - grad(x1, x2, method="bd", eps=eps))
    cd_errors[i] = np.linalg.norm(analytic_grad - grad(x1, x2, method="cd", eps=eps))

plt.figure(figsize=(10, 8))
plt.loglog(epsilons, fd_errors, label="Forward difference", color="red")
plt.loglog(epsilons, bd_errors, label="Backward difference", color="green")
plt.loglog(epsilons, cd_errors, label="Central difference", color="yellow")
plt.xlabel("$\epsilon$ (Log Scale)")
plt.ylabel("Error (Log Scale)")
plt.title("Error in gradient of $f(x_1, x_2)$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("plots/gradient_error_p2.png")

plt.figure(figsize=(10, 8))
plt.plot(epsilons, fd_errors, label="Forward difference", color="red")
plt.plot(epsilons, bd_errors, label="Backward difference", color="green")
plt.plot(epsilons, cd_errors, label="Central difference", color="yellow")
plt.xlim(1e-3, epsilons.max())
plt.xlabel("$\epsilon$ (Linear Scale)")
plt.ylabel("Error (Linear Scale)")
plt.title("Error in gradient of $f(x_1, x_2)$ (zoomed)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("plots/gradient_error_zoomed_p2.png")
