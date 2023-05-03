import pandas as pd
import math, csv
import matplotlib.pyplot as plt
import numpy as np
import inspect

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

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

DISTANCES = [0, 7.5, 22.5, 37.5, 52.5]

TITLE_SIZE = 18
AXIS_SIZE = 16
ERROR_INT = 40

###############################################################################
# EXERCISE A : Calculations
###############################################################################

def delta_t(tf):
    if tf > 0:
        return abs((tf - 23) * 0.0075)
    else:
        return abs((tf - 23) * 0.02)

def figure_1_plot():

    k = 8
    # xls = pd.read_excel('./Tables/Table.xlsx')
    T_min = list(reversed([-2.85, -4.21, -6.76, -11.87, -12.97, -19.18, -23.12]))
    s_min = list(reversed([0.99, 0.986, 0.977, 0.959, 0.994, 0.933, 0.919]))
    T_max = list(reversed([-1.92, -3.39, -6.04, 7.92, 6.95, 1.23, -2.21]))
    s_max = list(reversed([0.993, 0.988, 0.979, 1.556, 1.557, 1.559, 1.558]))
    T_max_err = list(map(delta_t, T_max))
    T_min_err = list(map(delta_t, T_min))

    plt.figure(figsize=(15,12))
    plt.title("T-s Diagram for Min and Max Evaporation Temperature", fontsize = TITLE_SIZE)        
    plt.plot(s_min[:k], T_min[:k], marker='o', markersize=7, 
                color=COLORS[0],  markeredgecolor="black", 
                markerfacecolor=COLORS[0])
    plt.plot(s_max[:k], T_max[:k], marker='o', markersize=7, 
                color=COLORS[1],  markeredgecolor="black", 
                markerfacecolor=COLORS[1])
    plt.errorbar(s_min[:k], T_min[:k], yerr=T_min_err[:k], 
                ls="None", color=COLORS[0], capsize=5)
    plt.errorbar(s_max[:k], T_max[:k], yerr=T_max_err[:k], 
                ls="None", color=COLORS[1], capsize=5)
    plt.xlabel('Entropy [kJ/kg*K]', fontsize = AXIS_SIZE)
    plt.ylabel('Temperature T_i [°C]', fontsize = AXIS_SIZE)
    plt.legend(['Min Temperature', 'Max Temperature'])
    plt.savefig('./Figure 2.png', bbox_inches='tight')

def figure_2_plot():

    k = 8
    # xls = pd.read_excel('./Tables/Table.xlsx')
    
    V = np.array([91, 86, 81, 75, 72, 60, 52])
    I = np.array([9.9, 9.4, 8.9, 8.1, 6.9, 6.6, 5.7])
    q = V*I
    T = np.array([-2.85, -4.21, -6.76, -11.87, -12.97, -19.18, -23.12])
    q_err = np.sqrt((1*V)**2 + (0.1*2)**2)


    plt.figure(figsize=(15,12))
    plt.title("Refrigeration Rate vs. Evaporator Temperature", fontsize = TITLE_SIZE)        
    plt.plot(T, q, marker='o', markersize=7, 
                color=COLORS[0],  markeredgecolor="black", 
                markerfacecolor=COLORS[0])
    plt.errorbar(T, q, yerr=q_err, 
                ls="None", color=COLORS[0], capsize=5)
    plt.xlabel('Temperature T_i [°C]', fontsize = AXIS_SIZE)
    plt.ylabel('Refrigeration Rate [W]', fontsize = AXIS_SIZE)
    plt.savefig('./Figure 3.png', bbox_inches='tight')

def figure_3_plot():

    k = 8
    # xls = pd.read_excel('./Tables/Table.xlsx')
    
    V = np.array([91, 86, 81, 75, 72, 60, 52])
    I = np.array([9.9, 9.4, 8.9, 8.1, 6.9, 6.6, 5.7])
    q = V*I
    COP = np.array([3.785270468, 3.971198157, 2.870081454, 2.249872277, 2.386178632, 1.191919241, 0.8079917681])
    COP_err = 1.5


    plt.figure(figsize=(15,12))
    plt.title("Coefficient of Performance vs. Refrigeration Rate [W]", fontsize = TITLE_SIZE)        
    plt.plot(q, COP, marker='o', markersize=7, 
                color=COLORS[0],  markeredgecolor="black", 
                markerfacecolor=COLORS[0])
    plt.errorbar(q, COP, yerr=COP_err, 
                ls="None", color=COLORS[0], capsize=5)
    plt.xlabel('Refrigeration Rate [W]', fontsize = AXIS_SIZE)
    plt.ylabel('Coefficient of Performance', fontsize = AXIS_SIZE)
    plt.savefig('./Figure 4.png', bbox_inches='tight')

def figure_4_plot():

    k = 8
    # xls = pd.read_excel('./Tables/Table.xlsx')
    T_min = [7.03, 60.49, 21.36, 21.76, -2.85, -1.92, 11.63, 19.57]
    s_min = [1.557, 1.54, 1.071, 1.073, 0.99, 0.993, 0.175, 0.29]
    T_max = [7.73, 69.03, 18.38, 20.05, -23.12, -2.21, 12.58, 26.58]
    s_max = [1.556, 1.537, 1.062, 1.067, 0.919, 1.558, 0.189, 0.39]
    T_max_err = list(map(delta_t, T_max))
    T_min_err = list(map(delta_t, T_min))

    plt.figure(figsize=(15,12))
    plt.title("T-s Diagram for Hilton and Ideal Cycle", fontsize = TITLE_SIZE)        
    plt.plot(s_min[:k], T_min[:k], marker='o', markersize=7, 
                color=COLORS[0],  markeredgecolor="black", 
                markerfacecolor=COLORS[0])
    plt.plot(s_max[:k], T_max[:k], marker='o', markersize=7, 
                color=COLORS[1],  markeredgecolor="black", 
                markerfacecolor=COLORS[1])
    plt.errorbar(s_min[:k], T_min[:k], yerr=T_min_err[:k], 
                ls="None", color=COLORS[0], capsize=5)
    plt.errorbar(s_max[:k], T_max[:k], yerr=T_max_err[:k], 
                ls="None", color=COLORS[1], capsize=5)
    plt.xlabel('Entropy [kJ/kg*K]', fontsize = AXIS_SIZE)
    plt.ylabel('Temperature T_i [°C]', fontsize = AXIS_SIZE)
    plt.legend(['Hilton Cycle', 'Ideal Cycle'])
    plt.savefig('./Figure 5.png', bbox_inches='tight')


def calculations():
    # exercise_a_calculations()
    figure_1_plot()
    figure_2_plot()
    figure_3_plot()
    figure_4_plot()

calculations()

print("Saved all figures.")

