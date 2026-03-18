import numpy as np
from sympy import*
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def L2_inner(f,g,beta,integral_end):
        return integrate.quad(lambda t: f(t)*g(t)*exp(-beta*t),0,integral_end)[0]


def e(n,alpha):
    return lambda t : np.polynomial.laguerre.Laguerre.basis(n)(t)* np.exp(-alpha*t)

def get_C(K, alpha, beta,M,integral_end, discrete_tau = False): #alpha must be float if discrete_tau is set to true
    C = []
    if not discrete_tau:
        for m in range(0,M+1):
            C.append(L2_inner(K,e(m,alpha),beta,integral_end))
    else:
        for m in range(0,M+1):
            C.append(e(m,alpha)(K)) #in case time delay is discrete, e.g. alpha = 5 which implies a discrete time delay of 5 seconds
    
    return np.array(C)

def get_B(m):
        B = np.ones(m+1)
        return B

def get_A(m,alpha, beta):
    A = -(np.tri(m+1) - np.diag(np.ones(m+1)-(alpha+beta)))
    return A

def simulate(alpha, beta ,M, t_span, y0, K, f, integral_end=oo, discrete_tau=False, plot_approx = true):    
    C= get_C(K,alpha, beta, M,integral_end, discrete_tau)
    A = get_A(M,alpha,beta)
    B = get_B(M)

    def ODE(t,y):
    # y[nx] = C @ y[nx +1:] # this defines z - might have to be moved to the bottom
        y[1] = C @ y[2:]
        ydot = np.zeros(M + 3)
        ydot[0] = f(y[0],y[1],t) # y[nx] = z
        ydot[2:] =  A @ y[2:] + B * y[0] # Zdot = ydot[nx+1:] and Z = y[nx+1:], r(y) = r(x) = x[rj]
        return ydot

    # Solve
    solution = solve_ivp(
        fun=lambda t, y: ODE(t, y),
        t_span=t_span,
        y0=y0,
        method="RK45", # ode45 equivalent
        rtol=1e-6,
        atol=1e-6
    )

    T = solution.t
    Y = solution.y.T


    if plot_approx:
        t = np.linspace(*t_span, 1000)


        Zs = np.array(Y[-1,2:]) # z_0 (t), z_1(t) ... where t is the last timestep in the simulation

        basis_functions = []
        for i in range(M+1):
            basis_functions.append(e(i,alpha)(t))
        
        basis_functions = np.array(basis_functions)

        v = basis_functions.T @ Zs.reshape(-1,1)

        plt.plot(t, v[::-1], label = "Approximation")
        

    # Plot
    plt.plot(T,Y[:,0],label=f" M = {M}") # Change the 0 to 1 to plot z instead.
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Numerical solutions")
    plt.legend()
    plt.grid(True)


def plot_kernel(K,alpha,beta, M,t_span,integral_end=oo, discrete_tau = False):
        tspan = np.linspace(*t_span,1000)
        C = np.array(get_C(K,alpha, beta, M,integral_end, discrete_tau))
        E = np.array([e(n,alpha)(tspan) for n in range(0,M+1)])
        approx = (C @ E)
        if not discrete_tau:
            plt.plot(tspan,K(tspan),label=r"$\alpha(t)$")
            plt.title(r"Kernel K(t) and a Laguerre approximation $\hat{\alpha}(t)$ with M=" + f"{M}")
        else:
            plt.axvline(1.0, linestyle='--', color='k', label=r"$\delta$(t-" + f"{alpha})")
            plt.title(r"Discrete time delay $K=\tau=$" + f"{K}" + r" and a Laguerre approximation $\hat{K}(t)$ with M=" + f"{M}")
        plt.plot(tspan,approx, label=r"$\hat{\alpha}(t)$")
        plt.ylabel("")
        plt.xlabel("t")

        plt.legend()
        plt.grid()