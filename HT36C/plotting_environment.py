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

def get_a_steady_states():
    xls = pd.read_excel('./Data/C5_HT36C_A.xls')
    # print(xls.columns)
    result = [[[] for _ in range(10)] for _ in range(4)]
    tube_no = xls['Number of Tubes']
    for i in range(len(xls['Number of Tubes'])):
        if np.isnan(tube_no[i]): continue
        for j in range(10):
            column = f'Temp T{j+1} [∞C]'
            result[int(tube_no[i]) - 1][j].append(xls[column][i])
    ss_result = [[] for _ in range(4)]
    for i in range(4):
        for j in range(10):
            ss_result[i].append(find_steady_state(result[i][j]))
    steady_states = [80, 30, 60, 60]

def exercise_a_calculations():
    xls = pd.read_excel('./Tables/Table 1.xlsx')
    a = np.array
    d_i = 0.0089
    L = 0.66*a([4, 3, 2, 1])
    A_i = d_i*np.pi*L
    T_h = 273.15 + 47.5
    T_c = 273.15 + 12.7
    c_p_c = 4182
    c_p_h = 4180
    rho_h = 1/(1.0114*10**-3)
    rho_c = 1/(1.0008*10**-3)
    m_dot_h = a(xls["V_hot [l/min]"])*rho_h/60000
    m_dot_c = a(xls["V_cold [l/min]"])*rho_c/60000
    T_c_i = a([12.8, 12.5, 12.6, 12.7])
    T_h_i = a(xls['T1 [°C]'])
    T_c_o = a(xls['T10 [°C]'])
    T_h_o = a([38.7, 41.1, 42.8, 44.8])
    q_c = -1*m_dot_c*c_p_c*(T_c_i - T_c_o)
    q_h = m_dot_h*c_p_h*(T_h_i - T_h_o)
    epsilon = q_h/(np.minimum(m_dot_c*c_p_c, m_dot_h*c_p_h)*(T_h_i - T_c_i))
    delta_T1 = T_h_i - T_c_o
    delta_T2 = T_h_o - T_c_i
    lmtd = (delta_T1 - delta_T2)/np.log(delta_T1/delta_T2)
    U_measured = q_h/(A_i*lmtd)
    delta_Delta_T = np.sqrt(0.1**2*2)
    delta_m_dot = 0.01*rho_h/60000
    delta_Q = np.sqrt((delta_m_dot*c_p_h*(T_h_i - T_h_o))**2+
                      (m_dot_h*c_p_h*delta_Delta_T)**2)
    delta_lmtd = np.sqrt(((delta_T1)*np.log(delta_T1/delta_T2) + delta_T2 - delta_T1)*delta_Delta_T/(delta_T1*np.log(delta_T1/delta_T2)**2)**2 +
                         ((-delta_T2)*np.log(delta_T1/delta_T2) + delta_T1 - delta_T2)*delta_Delta_T/(delta_T2*np.log(delta_T1/delta_T2)**2)**2)
    delta_U_measured = np.sqrt((delta_Q/(A_i*lmtd))**2 + (-q_h*delta_lmtd/(A_i*lmtd**2))**2)
    mu = 567*10**-6
    Re = 4*m_dot_h/(np.pi*d_i*mu)
    k = 641*10**-3
    Pr = 0.894
    Nu = 0.023*Re**0.8*Pr**0.3
    h_i = k*Nu/d_i
    d_o = 0.014
    A_o = d_o*np.pi*L
    Re = 4*m_dot_h/(np.pi*d_o*mu)
    Nu = 0.023*Re**0.8*Pr**0.3
    h_o = k*Nu/d_o
    k_steel = 15.2
    R_tube = np.log(d_o/d_i)/(2*np.pi*L*k_steel)
    U_predicted = (1/(h_i*A_i) + R_tube + 1/(h_o*A_o))**-1/A_i
    U_diff = abs(U_measured - U_predicted)/(U_predicted)*100
    
    save(m_dot_h, m_dot_c, T_c_i, T_h_i, T_c_o, T_h_o, q_c, q_h, epsilon, 
         lmtd, U_measured, delta_U_measured, h_i, h_o, R_tube, U_predicted, U_diff)
    print_values(2)
    VALUES.clear()
    save(rho_c, rho_h, c_p_c, c_p_h, mu, Pr, k, k_steel, length=4)
    print_values(3)

def get_b_steady_states():
    xls = pd.read_excel('./Data/C5_HT36C_B1.xls')
    # print(xls.columns)
    result = [[[] for _ in range(10)] for _ in range(4)]
    tube_no = xls['Number of Tubes']
    for i in range(len(xls['Number of Tubes'])):
        if np.isnan(tube_no[i]): continue
        for j in range(10):
            column = f'Temp T{j+1} [∞C]'
            result[int(tube_no[i]) - 1][j].append(xls[column][i])
    ss_result = [[] for _ in range(4)]
    for i in range(4):
        for j in range(10):
            ss_result[i].append(find_steady_state(result[i][j]))
    steady_states = [80, 30, 60, 60]

def exercise_b_calculations():
    xls = pd.read_excel('./Tables/Table 4.xlsx')
    a = np.array
    d_i = 0.0089
    L = 0.66*4
    A_i = d_i*np.pi*L
    T_h = 273.15 + 47.5
    T_c = 273.15 + 12.7
    c_p_c = 4182
    c_p_h = 4180
    rho_h = 1/(1.0114*10**-3)
    rho_c = 1/(1.0008*10**-3)
    m_dot_h = a(xls["V_hot [l/min]"])*rho_h/60000
    m_dot_c = a(xls["V_cold [l/min]"])*rho_c/60000
    T_c_i = a(xls['T6 [°C]'])
    T_h_i = a(xls['T1 [°C]'])
    T_c_o = a(xls['T10 [°C]'])
    T_h_o = a(xls['T5 [°C]'])
    q_c = -1*m_dot_c*c_p_c*(T_c_i - T_c_o)
    q_h = m_dot_h*c_p_h*(T_h_i - T_h_o)
    epsilon = q_h/(np.minimum(m_dot_c*c_p_c, m_dot_h*c_p_h)*(T_h_i - T_c_i))
    delta_T1 = T_h_i - T_c_i
    delta_T2 = T_h_o - T_c_o
    lmtd = (delta_T1 - delta_T2)/np.log(delta_T1/delta_T2)
    U_measured = q_h/(A_i*lmtd)
    delta_Delta_T = np.sqrt(0.1**2*2)
    delta_m_dot = 0.01*rho_h/60000
    delta_Q = np.sqrt((delta_m_dot*c_p_h*(T_h_i - T_h_o))**2+
                      (m_dot_h*c_p_h*delta_Delta_T)**2)
    delta_lmtd = np.sqrt(((delta_T1)*np.log(delta_T1/delta_T2) + delta_T2 - delta_T1)*delta_Delta_T/(delta_T1*np.log(delta_T1/delta_T2)**2)**2 +
                         ((-delta_T2)*np.log(delta_T1/delta_T2) + delta_T1 - delta_T2)*delta_Delta_T/(delta_T2*np.log(delta_T1/delta_T2)**2)**2)
    delta_U_measured = np.sqrt((delta_Q/(A_i*lmtd))**2 + (-q_h*delta_lmtd/(A_i*lmtd**2))**2)
    
    save(m_dot_h, m_dot_c, T_c_i, T_h_i, T_c_o, T_h_o, q_c, q_h, epsilon, 
         lmtd, U_measured, delta_U_measured)
    print_values(5)

def get_c_steady_states():
    xls = pd.read_excel('./Data/C5_HT36C_C1.xls')
    ss_result = []
    for j in range(10):
        ss_result.append(find_steady_state(xls[f"Temp T{j+1} [∞C]"]))
    # print(ss_result)
    # ss1 = 35
    # ss2 = 20

get_c_steady_states()

def exercise_c_calculations():
    a = np.array
    d_i = 0.0089
    L = 0.66*4
    A_i = d_i*np.pi*L
    c_p_c = 4182
    c_p_h = 4180
    rho_h = 1/(1.01*10**-3)
    rho_c = 1/(1.0002*10**-3)
    m_dot_h = a([1.01, 0.48])*rho_h/60000
    m_dot_c = a([1.01, 0.97])*rho_c/60000
    T_c_i = a([13, 13.6])
    T_h_i = a([43.6, 44.5])
    T_c_o = a([26.8, 22.8])
    T_h_o = a([30.9, 29.1])
    q_c = -1*m_dot_c*c_p_c*(T_c_i - T_c_o)
    q_h = m_dot_h*c_p_h*(T_h_i - T_h_o)
    epsilon = q_h/(np.minimum(m_dot_c*c_p_c, m_dot_h*c_p_h)*(T_h_i - T_c_i))
    delta_T1 = T_h_i - T_c_i
    delta_T2 = T_h_o - T_c_o
    lmtd = (delta_T1 - delta_T2)/np.log(delta_T1/delta_T2)
    U_measured = q_h/(A_i*lmtd)
    delta_Delta_T = np.sqrt(0.1**2*2)
    delta_m_dot = 0.01*rho_h/60000
    delta_Q = np.sqrt((delta_m_dot*c_p_h*(T_h_i - T_h_o))**2+
                      (m_dot_h*c_p_h*delta_Delta_T)**2)
    delta_lmtd = np.sqrt(((delta_T1)*np.log(delta_T1/delta_T2) + delta_T2 - delta_T1)*delta_Delta_T/(delta_T1*np.log(delta_T1/delta_T2)**2)**2 +
                         ((-delta_T2)*np.log(delta_T1/delta_T2) + delta_T1 - delta_T2)*delta_Delta_T/(delta_T2*np.log(delta_T1/delta_T2)**2)**2)
    delta_U_measured = np.sqrt((delta_Q/(A_i*lmtd))**2 + (-q_h*delta_lmtd/(A_i*lmtd**2))**2)
    
    save(m_dot_h, m_dot_c, T_c_i, T_h_i, T_c_o, T_h_o, q_c, q_h, epsilon, 
         lmtd, U_measured, delta_U_measured)
    print_values(7)

def get_d_steady_states():
    xls = pd.read_excel('./Data/C5_HT36C_D4.xls')
    ss_result = []
    for j in range(10):
        ss_result.append(find_steady_state(xls[f"Temp T{j+1} [∞C]"]))
    # print(ss_result)
    # ss1 = 37
    # ss2 = 29
    # ss3 = 28

get_d_steady_states()

def exercise_d_calculations():
    xls = pd.read_excel('./Tables/Table 8.xlsx')
    a = np.array
    d_i = 0.0089
    L = 0.66*4
    A_i = d_i*np.pi*L
    c_p_c = 4182
    c_p_h = 4180
    rho_h = 1/(1.012*10**-3)
    rho_c = 1/(1.0002*10**-3)
    m_dot_h = a([3, 2, 1, 0.51])*rho_h/60000
    m_dot_c = a([0.95, 0.94, 1.07, 1.1])*rho_c/60000
    T_c_i = a(xls['T6 [°C]'])
    T_h_i = a(xls['T1 [°C]'])
    T_c_o = a(xls['T10 [°C]'])
    T_h_o = a(xls['T5 [°C]'])
    q_c = -m_dot_c*c_p_c*(T_c_i - T_c_o)
    q_h = m_dot_h*c_p_h*(T_h_i - T_h_o)
    epsilon = q_h/(np.minimum(m_dot_c*c_p_c, m_dot_h*c_p_h)*(T_h_i - T_c_i))
    delta_T1 = T_h_i - T_c_i
    delta_T2 = T_h_o - T_c_o
    lmtd = (delta_T1 - delta_T2)/np.log(delta_T1/delta_T2)
    U_measured = q_h/(A_i*lmtd)
    delta_Delta_T = np.sqrt(0.1**2*2)
    delta_m_dot = 0.01*rho_h/60000
    delta_Q = np.sqrt((delta_m_dot*c_p_h*(T_h_i - T_h_o))**2+
                      (m_dot_h*c_p_h*delta_Delta_T)**2)
    delta_lmtd = np.sqrt(((delta_T1)*np.log(delta_T1/delta_T2) + delta_T2 - delta_T1)*delta_Delta_T/(delta_T1*np.log(delta_T1/delta_T2)**2)**2 +
                         ((-delta_T2)*np.log(delta_T1/delta_T2) + delta_T1 - delta_T2)*delta_Delta_T/(delta_T2*np.log(delta_T1/delta_T2)**2)**2)
    delta_U_measured = np.sqrt((delta_Q/(A_i*lmtd))**2 + (-q_h*delta_lmtd/(A_i*lmtd**2))**2)

    
    save(m_dot_h, m_dot_c, T_c_i, T_h_i, T_c_o, T_h_o, q_c, q_h, epsilon, 
         lmtd, U_measured, delta_U_measured)
    print_values(9)
    
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
    exercise_a_calculations()
    VALUES.clear()
    exercise_b_calculations()
    VALUES.clear()
    exercise_c_calculations()
    VALUES.clear()
    exercise_d_calculations()
    figure_1_plot()
    return

calculations()

print("Saved all figures.")

