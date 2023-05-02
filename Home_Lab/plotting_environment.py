import pandas as pd
import math, csv
import matplotlib.pyplot as plt
import numpy as np
import inspect
from math import sqrt, exp, pi
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

###############################################################################
# HELPER FUNCTIONS & CONSTANTS
###############################################################################

x, y, z = 1, 2, 3

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def save(*args, length = None):
    if length == None:
        length = len(args[0])
    for arg in args:
        if not isinstance(arg, np.ndarray):
            VALUES[retrieve_name(arg)] = np.array([arg]*length)
        else:
            VALUES[retrieve_name(arg)] = arg

# returns the time value from a given sample number
def sample_to_time(n: int) -> int:
    return (n-1) * 10

def get_average(data: list[float], ss_start:int):
    return sum(data[ss_start:]) / len(data[ss_start:])

def temp_err(T : float) -> float:
    return (2.2 if T*0.0075 < 2.2 else T*0.0075)/2

def linear_interpolation(x1 : float, y1 : float, x2 : float, 
                         y2 : float, x3 : float) -> float:
    assert(x1 <= x3 <= x2)
    return y1 + (y2 - y1) * (x3 - x1) / (x2 - x1)


def find_steady_state(data : list[float]) ->int:
    assert(len(data) > 10)
    offset = 15
    init = (data[offset + 1] - data[1]) / offset
    for i in range(1, len(data) - offset):
        slope = (data[i+offset] - data[i]) / offset
        if abs(slope) <= abs(init) * 0.15:
            # if i < int(0.7 * len(data)):
            #     continue
            return i
    return None

def get_average_save_0(column):
    assert(len(column) % 5 == 0)
    result = []
    for i in range(len(column)):
        if i % 5 == 0:
            if len(result) > 0:
                result[-1] /= 5
            result.append(0)
        result[-1] += column[i]
    result[-1] /= 5 
    return result

def get_average_save(xls, name):
    column = list(xls[name])
    assert(len(column) % 5 == 0)
    result = []
    for i in range(len(column)):
        if i % 5 == 0:
            if len(result) > 0:
                result[-1] /= 5
            result.append(0)
        result[-1] += column[i]
    result[-1] /= 5 
    VALUES[name] = result
    return result

def print_table(table_no : int, *args) -> None:
    result = []
    for row in range(len(args)):
        result.append(row)
    rows, cols = len(result), len(result[0])
    rotated = [[0] * cols for _ in range(rows)]
    for i in range(len(result)):
        for j in range(len(result[0])):
            rotated[i][j] = result[j][i]
    with open(f"Table {table_no}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rotated)
        
VALUES = dict()

def print_values(c : str) -> None:
    result = []
    for key in VALUES:
        result.append([key] + list(VALUES[key]))
    rows = len(result)
    cols = max([len(result[i]) for i in range(rows)])
    rotated = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            rotated[j][i] = result[i][j]
    with open(f"Values Table {c}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rotated)

def make_list(s):
    return list(map(float,list(s.splitlines())))

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

DISTANCES = [0, 7.5, 22.5, 37.5, 52.5]

TITLE_SIZE = 18
AXIS_SIZE = 16
ERROR_INT = 40

###############################################################################
# EXERCISE A : Calculations
###############################################################################

import numpy as np
from math import sqrt
import random
from sklearn.metrics import mean_squared_error

def figure_plot():

    xls = pd.read_excel('./Home lab.xlsx')

    # Initialize data
    # T_inf = (72 - 32) * (5 / 9)
    rho = 1000
    c = 4184
    V = 0.00118294
    g = 9.81
    T_inf = 294.95
    H = 0.085725
    delta_H = 0.005
    d_p = 0.2413
    delta_d_p = 0.05
    A_p = np.pi * d_p * H
    A_w = (np.pi * d_p ** 2) / 4
    q_b = 3000

    time = np.array([0, 30, 60, 90, 120, 150, 180, 200])
    
    T_w = np.array([23.5, 27.3, 32.0, 39.8, 48.9, 56.5, 64.8, 70.8])
    T_w_abs = np.array([(random.random(0, 0.1)) + T_w[i] for i in range(len(T_w))])
    print(T_w_abs)
    delta_T = 1.25 * np.ones(len(T_w))
    T_f = np.array(xls['Film Temp (K)'][10:])
    delta_T_f = 0.5
    beta = np.array(xls['Beta'][10:])
    delta_beta = 1/T_f**2*delta_T_f
    v = np.array(xls['Viscosity'][10:])
    alpha = np.array(xls['alpha'][10:])
    Pr = np.array(xls['Pr'][10:])
    Ra_L = np.array(xls['Ra_L'][10:])
    term1 = (3 * g * beta * (T_inf - T_w) * H**2) / (v * alpha) * delta_H
    term2 = (g * (T_inf - T_w) * H**3) / (v * alpha) * delta_beta
    term3 = (2 * g * beta * H**3) / (v * alpha) * delta_T
    delta_Ra_L = np.sqrt(term1**2 + term2**2 + term3**2)
    Nu_L = np.array(xls['Nu_L'][10:])
    term = (1 + (0.492 / Pr)**(9 / 16))**(4 / 9)
    delta_Nu_L = (0.67 / 4) * (Ra_L**(3 / 4) * term)**(-1) * delta_Ra_L
    k = np.array(xls['k'][10:])
    h_L = np.array(xls['h_L'][10:])
    term1 = (Nu_L * k / H**2) * delta_H
    term2 = (k / H) * delta_Nu_L
    delta_h_H = np.sqrt(term1**2 + term2**2)
    A_L = np.array(xls['A_L'][10:])
    term1 = (np.pi * d_p * delta_H)
    term2 = (np.pi * H * delta_d_p)
    delta_A_H = np.sqrt(term1**2 + term2**2)
    q_L = np.array(xls['q_L'][10:])
    term1 = (h_L * (T_inf - T_w) * delta_A_H)
    term2 = (2 * h_L * A_L * delta_T)
    delta_q_pot_surr = np.sqrt(term1**2 + term2**2)
    Ra_D = np.array(xls['Ra_D'][10:])
    term1 = (3 * g * beta * (T_inf - T_w) * d_p**2) / (v * alpha) * delta_H
    term2 = (g * (T_inf - T_w) * d_p**3) / (v * alpha) * delta_beta
    term3 = (2 * g * beta * d_p**3) / (v * alpha) * delta_T
    delta_Ra_D = np.sqrt(term1**2 + term2**2 + term3**2)
    term1 = (Nu_D * k / d_p**2) * delta_d_p
    term2 = (k / d_p) * delta_Nu_D
    delta_h_D = np.sqrt(term1**2 + term2**2)
    # if 1e4 <= Ra_D <= 1e7:
    #     delta_Nu_D = 0.135 * Ra_D**(-3 / 4) * delta_Ra_D
    # elif 1e7 <= Ra_D <= 1e11:
    #     delta_Nu_D = 0.05 * Ra_D**(-2 / 3) * delta_Ra_D
    Nu_D = np.array(xls['Nu_D'][10:])
    h_D = np.array(xls['h_D'][10:])
    A_D = np.array(xls['A_D'][10:])
    q_D = np.array(xls['q_D'][10:])

    # Question 4
    h_L = np.array(xls['h_L'][10:])
    avg_h_L = np.mean(h_L)
    h_D = np.array(xls['h_D'][10:])
    avg_h_D = np.mean(h_D)

    a = (1 / (rho * V * c)) * ((avg_h_L * A_p) + (avg_h_D * A_w))
    b = lambda q: q / (rho * V * c)
    def Tw(q, time, T_inf, T_0, a):
        return T_inf + (T_0 - T_inf) * (np.exp(-a * time) + ((q / a) / (T_0 - T_inf)) * (1 - np.exp(-a * time)))

    def mse_function(q, time, T_exp, T_inf, T_0, a):
        T_pred = Tw(q, time, T_inf, T_0, a)
        return mean_squared_error(T_exp, T_pred)
    

    delta_Tw = np.array([0.5, 0.6, 0.3, 0.7, 0.4])

    initial_guess = 1000  # You can adjust the initial guess for the optimization algorithm
    result = minimize(mse_function, initial_guess, args=(time, T, T_inf, T[0], a))
    q_in = result
    print(q_in)

    # # 5c
    # plt.figure(figsize=(10, 6))
    # plt.errorbar(time, T, yerr=delta_T, fmt='b-o', linewidth=2, label='Experimental Temperatures')
    # plt.errorbar(time, T_calc, yerr=delta_Tw, fmt='r-', linewidth=2, label='Fitted T_{water}(t)')
    # plt.title('Fitted T_{water}(t) Using Lumped Capacitance Model')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Temperature (Celsius)')
    # plt.legend(loc='southeast', fontsize=12)
    # plt.show()

    # # 6
    # q_pot = np.array([7.921098336, 130.5416634, 262.9208276, 408.7864639, 571.6711259])
    # delta_q_pot = np.array([6.075228836, 12.94552531, 19.04217495, 26.06098468, 33.97184359])
    # q_wat = np.array([0.1138276261, 1.984882112, 4.072355276, 6.479416432, 9.137644101])
    # delta_q_wat = np.array([0.09202101499, 0.5035792617, 0.9548329459, 1.461212602, 2.008403874])
    # q_watabs = np.array([0, 175.5170786, 346.6164356, 513.4092641, 676.0039582])
    # delta_q_watabs = np.array([5, 5, 5, 5, 5])

    # plt.figure(figsize=(10, 6))
    # plt.errorbar(time, q_b * np.ones(len(time)), yerr=0 * np.ones(len(time)), fmt='b', linewidth=2, label='q_{burner}')
    # plt.errorbar(time, q_in * np.ones(len(time)), yerr=10 * np.ones(len(time)), fmt='r-', linewidth=2, label='q_{in}')
    # plt.errorbar(time, q_pot, yerr=delta_q_pot, linewidth=2, label='q_{pot,surr}')
    # plt.errorbar(time, q_wat, yerr=delta_q_wat, linewidth=2, label='q_{wat,surr}')
    # plt.errorbar(time, q_watabs, yerr=delta_q_watabs, linewidth=2, label='q_{wat,abs}')
    # plt.title('Heat Flows vs Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Watts (W)')
    # plt.legend(loc='best', fontsize=12)
    # plt.show()
    
def figure_1_plot():

    xls = pd.read_excel('./Tables/Table 4 Plot.xlsx')

    cf_dist = xls['Distance'][:5]
    cf_T_h = xls['Counter Flow'][:5]
    cf_T_c = xls['Counter Flow'][5:]
    pf_dist = xls['Distance'][:5]
    pf_T_h = xls['Parallel Flow'][:5]
    pf_T_c = xls['Parallel Flow'][5:]
    delta_T = 1

    plt.figure(figsize=(15,12))
    plt.title("Counter Flow Temperature T_i [°C]vs. Distance d [m]", fontsize = TITLE_SIZE)
    for i in range(2):
        curve = (cf_T_h, cf_T_c)[i]
        plt.plot(cf_dist, curve, marker='o', markersize=7, 
                    color=COLORS[i],  markeredgecolor="black", 
                    markerfacecolor=COLORS[i])
        plt.errorbar(cf_dist, curve, yerr=delta_T, 
                    ls="None", color=COLORS[i], capsize=5)
    plt.xlabel('Distance d [m]', fontsize = AXIS_SIZE)
    plt.ylabel('Temperature T_i [°C]', fontsize = AXIS_SIZE)
    plt.legend(['Hot Water Flow', 'Cold Water Flow'])
    plt.savefig('./Figure 1.png', bbox_inches='tight')

    plt.figure(figsize=(15,12))
    plt.title("Parallel Flow Temperature T_i [°C]vs. Distance d [m]", fontsize = TITLE_SIZE)
    for i in range(2):
        curve = (pf_T_h, pf_T_c)[i]
        plt.plot(pf_dist, curve, marker='o', markersize=7, 
                    color=COLORS[i],  markeredgecolor="black", 
                    markerfacecolor=COLORS[i])
        plt.errorbar(pf_dist, curve, yerr=delta_T, 
                    ls="None", color=COLORS[i], capsize=5)
    plt.xlabel('Distance d [m]', fontsize = AXIS_SIZE)
    plt.ylabel('Temperature T_i [°C]', fontsize = AXIS_SIZE)
    plt.legend(['Hot Water Flow', 'Cold Water Flow'])
    plt.savefig('./Figure 2.png', bbox_inches='tight')

def calculations():
    figure_plot()
    # figure_1_plot()
    return

calculations()

print("Saved all figures.")

