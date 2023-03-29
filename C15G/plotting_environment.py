import pandas as pd
import math, csv
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# HELPER FUNCTIONS & CONSTANTS
###############################################################################

sin = lambda x: np.sin(np.radians(x))
cos = lambda x: np.cos(np.radians(x))
tan = lambda x: np.tan(np.radians(x))

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
    assert(y1 <= y2)
    return y1 + (y2 - y1) * (x3 - x1) / (x2 - x1)

def find_steady_state(data : list[float]) ->int:
    assert(len(data) > 10)
    init = (data[13] - data[3]) / 10
    for i in range(0, len(data) - 10):
        slope = (data[i+10] - data[i]) / 10
        if abs(slope) < abs(init) * 0.1:
            return i
    return None

def print_table(table_no : int, *args) -> None:
    result = []
    for row in range(len(args)):
        result.append(row)
    with open(f"Table {table_no}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)
        
VALUES = dict()

def print_values(c : str) -> None:
    result = []
    for key in VALUES:
        result.append([key, VALUES[key]])
    with open(f"Values Table {c}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(result)

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

DISTANCES = [0, 7.5, 22.5, 37.5, 52.5]

TITLE_SIZE = 15
AXIS_SIZE = 13

NUM_ANGLES = 5
NUM_VELS = 5
NUM_RUNS = 5
NUM_TUBES = 10

###############################################################################
# EXERCISE A : THERMOCOUPLE TEMPERATURE VS TIME
###############################################################################


###############################################################################
# EXERCISE A : Calculations
###############################################################################

def exercise_a_calculations():

    xls = pd.read_excel('./C15G Data.xls')

def get_run(angle : int, vel : int, run : int) -> int:
    return angle * 25 + vel * 5 + run

###############################################################################
# EXERCISE G : Figure 1 Plot
###############################################################################

def figure_1_plot():

    xls = pd.read_excel('./C15G Data.xls')

    l = np.array(xls["Lift Force L [N]"])

    l_avg = []
    v_avg = [25, 20, 15, 10, 5]

    for angle in range(NUM_ANGLES):
        l_avg.append([])
        for vel in range(NUM_VELS):
            curr_L_avg = 0
            for run in range(NUM_RUNS):
                curr_L_avg += l[get_run(angle, vel, run)] / 5
            l_avg[-1].append(curr_L_avg)
        l_avg[-1] = np.array(l_avg[-1])

    VALUES["l"] = l_avg

    plt.figure(figsize=(25,10))
    plt.title("Lift Force L [N] vs. Air Velocity v [m/s]", fontsize = TITLE_SIZE)

    angles = np.array([0, 5, 10, 20, 30])
    assert(len(angles) == len(l_avg))
    delta_L = []
    L = []
    for i in range(len(l_avg)):
        L_runs = l_avg[i] / cos(angles[i])
        L_err = np.sqrt((0.1 / cos(angles))**2 + (l[i]*sin(angles)*math.radians(0.1)/cos(angles)**2)**2)
        
        L.append(L_runs)
        delta_L.append(L_err)

        plt.plot(v_avg, L_runs, marker='o', markersize=7, color=COLORS[i], 
            markeredgecolor="black", markerfacecolor=COLORS[i])
        plt.errorbar(v_avg, L_runs, yerr=L_err, ls="None", color=COLORS[i], capsize=5)

    VALUES["L"] = L
    VALUES['delta_L'] = delta_L    
    
    plt.xlabel('Air Velocity v [m/s]', fontsize = AXIS_SIZE)
    plt.ylabel('Lift Force L [N]', fontsize = AXIS_SIZE)
    plt.legend(['0°', '5°', '10°', '20°', '30°'])
    plt.savefig('./Figure 1.png', bbox_inches='tight')

figure_1_plot()

###############################################################################
# EXERCISE G : Figure 2 Plot
###############################################################################

def figure_2_plot():

    xls = pd.read_excel('./C15G Data.xls')

    l = np.array(xls["Lift Force L [N]"])

    # airfoil dimensions
    chord = 0.061
    span = 0.14
    S = chord * span

    rho_air = 1.163
    v_avg = np.array([25, 20, 15, 10, 5])

    C_L = []
    L = VALUES["L"]
    for i in range(NUM_ANGLES):
        C_L.append((2 * L[i]) / (rho_air * v_avg**2 * S))

    VALUES["C_L"] = C_L

    plt.figure(figsize=(25,10))
    plt.title("Lift Coefficient C_L vs. Air Velocity v [m/s]", fontsize = TITLE_SIZE)

    angles = np.array([0, 5, 10, 20, 30])
    delta_rho_air = 0.001
    delta_v_avg = 0.1

    for i in range(len(C_L)):
        # delta_L = VALUES["L"][i]
        delta_L = 0.01
        delta_C_L = np.sqrt(((2/(rho_air*(v_avg**2)*S))*delta_L)**2 +
                    ((-2*L[i]/((rho_air**2)*(v_avg**2)*S))*delta_rho_air)**2 +
                    ((-4*L[i]/(rho_air*(v_avg**3)*S))*delta_v_avg)**2)
        plt.plot(v_avg, C_L[i], marker='o', markersize=7, color=COLORS[i], 
            markeredgecolor="black", markerfacecolor=COLORS[i])
        plt.errorbar(v_avg, C_L[i], yerr=delta_C_L, ls="None", color=COLORS[i], capsize=5)
        
    VALUES["delta_C_L"] = delta_C_L

    plt.xlabel('Air Velocity v [m/s]', fontsize = AXIS_SIZE)
    plt.ylabel('Lift Coefficient C_L', fontsize = AXIS_SIZE)
    plt.legend(['0°', '5°', '10°', '20°', '30°'])
    plt.savefig('./Figure 2.png', bbox_inches='tight')

figure_2_plot()

###############################################################################
# EXERCISE G : Figure 3 Plot
###############################################################################

def figure_3_plot():

    xls = pd.read_excel('./C15G Data.xls')

    d = np.array(xls["Drag Force D [N]"])

    D_avg = []
    v_avg = [25, 20, 15, 10, 5]

    for angle in range(NUM_ANGLES):
        D_avg.append([])
        for vel in range(NUM_VELS):
            curr_D_avg = 0
            for run in range(NUM_RUNS):
                curr_D_avg += d[get_run(angle, vel, run)] / 5
            D_avg[-1].append(curr_D_avg)

    plt.figure(figsize=(25,10))
    plt.title("Drag Force D [N] vs. Air Velocity v [m/s]", fontsize = TITLE_SIZE)

    angles = np.array([0, 5, 10, 20, 30])
    assert(len(angles) == len(D_avg))
    delta_D = []
    for i in range(len(D_avg)):
        delta_D = 0.1
        plt.plot(v_avg, D_avg[i], marker='o', markersize=7, color=COLORS[i], 
            markeredgecolor="black", markerfacecolor=COLORS[i])
        plt.errorbar(v_avg, D_avg[i], yerr=delta_D, ls="None", color=COLORS[i], capsize=5)

    VALUES["D"] = D_avg
    VALUES['delta_D'] = delta_D 
    
    plt.xlabel('Air Velocity v [m/s]', fontsize = AXIS_SIZE)
    plt.ylabel('Drag Force D [N]', fontsize = AXIS_SIZE)
    plt.legend(['0°', '5°', '10°', '20°', '30°'])
    plt.savefig('./Figure 3.png', bbox_inches='tight')

figure_3_plot()

###############################################################################
# EXERCISE G : Figure 4 Plot
###############################################################################

def figure_4_plot():

    xls = pd.read_excel('./C15G Data.xls')

    d = np.array(xls["Drag Force D [N]"])

    # airfoil dimensions
    chord = 0.061
    span = 0.14
    S = chord * span

    alpha = np.array([0, 5, 10, 20, 30])
    rho_air = 1.163

    D = []
    v_avg = np.array([25, 20, 15, 10, 5])

    for angle in range(NUM_ANGLES):
        D.append([])
        for vel in range(NUM_VELS):
            curr_D_avg = 0
            for run in range(NUM_RUNS):
                curr_D_avg += d[get_run(angle, vel, run)] / 5
            D[-1].append(curr_D_avg)

    C_D_avg = []
    C_L = VALUES["C_L"]
    l = VALUES["l"]

    for i in range(NUM_ANGLES):
        k = 2 * np.array(l[i]) * tan(alpha[i]) / (rho_air * v_avg**2 * S * C_L[i])
        C_DL = k * C_L[i] ** 2
        C_D0 = 2 * np.array(D[i]) * tan(alpha[i]) / (rho_air * v_avg**2 * S * C_L[i])
        C_D = C_D0 + C_DL
        L = VALUES['L'][i]
        C_D_avg.append(C_D)

    plt.figure(figsize=(25,10))
    plt.title("Drag Coefficient C_D vs. Air Velocity v [m/s]", fontsize = TITLE_SIZE)

    angles = np.array([0, 5, 10, 20, 30])
    delta_rho_air = 0.001
    delta_v_avg = 0.1

    for i in range(len(C_D_avg)):
        delta_D = 0.01
        L = VALUES['L'][i]
        delta_C_D = np.sqrt(((2/(rho_air*(v_avg**2)*S))*delta_D)**2 +
                    ((-2*L/((rho_air**2)*(v_avg**2)*S))*delta_rho_air)**2 +
                    ((-4*L/(rho_air*(v_avg**3)*S))*delta_v_avg)**2)
        plt.plot(v_avg, C_D_avg[i], marker='o', markersize=7, color=COLORS[i], 
            markeredgecolor="black", markerfacecolor=COLORS[i])
        plt.errorbar(v_avg, C_D_avg[i], yerr=delta_C_D, ls="None", color=COLORS[i], capsize=5)
        
    plt.xlabel('Air Velocity v [m/s]', fontsize = AXIS_SIZE)
    plt.ylabel('Drag Coefficient C_D', fontsize = AXIS_SIZE)
    plt.legend(['0°', '5°', '10°', '20°', '30°'])
    plt.savefig('./Figure 4.png', bbox_inches='tight')

figure_4_plot()

###############################################################################
# EXERCISE C : Calculations
###############################################################################

def figure_5_plot():

    xls = pd.read_excel('./C15G Data.xls')

    d = np.array(xls["Drag Force D [N]"])

    # airfoil dimensions
    chord = 0.061
    span = 0.14
    S = chord * span

    alpha = np.array([0, 5, 10, 20, 30])
    rho_air = 1.163

    D = []
    v_avg = np.array([25, 20, 15, 10, 5])

    for angle in range(NUM_ANGLES):
        D.append([])
        for vel in range(NUM_VELS):
            curr_D_avg = 0
            for run in range(NUM_RUNS):
                curr_D_avg += d[get_run(angle, vel, run)] / 5
            D[-1].append(curr_D_avg)

    C_DL_avg = []
    k_avg = []
    C_L = VALUES["C_L"]
    l = VALUES["l"]

    tan = lambda x: np.tan(np.radians(x))
    for i in range(NUM_ANGLES):
        k = 2 * l[i] * tan(alpha[i]) / (rho_air * v_avg**2 * S * C_L[i]**2)
        k_avg.append(k)
        C_DL = k * C_L[i]**2
        C_DL_avg.append(C_DL)

    plt.figure(figsize=(25,10))
    plt.title("Drag Coefficient C_DL vs. Air Velocity v [m/s]", fontsize = TITLE_SIZE)

    delta_rho_air = 0.001
    delta_v_avg = 0.1
    delta_l = 0.1
    delta_alpha = 0.1
    delta_D = 0.01

    for i in range(len(C_DL_avg)):
        # print(C_DL_runs)
        delta_C_L = VALUES['delta_C_L'][i]
        term1 = ((2 * tan(alpha[i]) * delta_l) / (rho_air * v_avg**2 * S * (C_L[i]**2)))**2
        term2 = ((2 * delta_alpha * l[i]) / (rho_air * v_avg**2 * S * (C_L[i]**2) * cos(alpha[i])**2))**2
        term3 = ((-2 * l[i] * tan(alpha[i]) * delta_rho_air) / ((rho_air**2) * v_avg**2 * S * (C_L[i]**2)))**2
        term4 = ((-4 * l[i] * tan(alpha[i]) * delta_v_avg**2) / (rho_air * (v_avg**3) * S * (C_L[i]**2)))**2
        term5 = ((-4 * l[i] * tan(alpha[i]) * delta_C_L) / (rho_air * v_avg**2 * S * (C_L[i]**3)))**2
        delta_k = np.sqrt(term1 + term2 + term3 + term4 + term5)
        delta_C_DL = np.sqrt(((C_L[i]**2 * delta_k)**2)) + ((2 * k[i] * C_L[i] * delta_C_L)**2)
        plt.plot(v_avg, C_DL_avg[i], marker='o', markersize=7, color=COLORS[i], 
            markeredgecolor="black", markerfacecolor=COLORS[i])
        plt.errorbar(v_avg, C_DL_avg[i], yerr=delta_C_DL, ls="None", color=COLORS[i], capsize=5)
        
    plt.xlabel('Air Velocity v [m/s]', fontsize = AXIS_SIZE)
    plt.ylabel('Drag Coefficient C_DL', fontsize = AXIS_SIZE)
    plt.legend(['0°', '5°', '10°', '20°', '30°'])
    plt.savefig('./Figure 5.png', bbox_inches='tight')

figure_5_plot()

###############################################################################
# EXERCISE C : THERMOCOUPLE TEMPERATURE VS TIME
###############################################################################

def figure_6_plot():

    v_table_1 = [[4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11],
               [4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11, 4.11],
               [4.11, 4.11, 4.11, 4.11, 0, 0, 0, 4.11, 4.11, 4.11],
               [4.11, 0, 0, 0, 0, 0, 0, 0, 0, 4.11],
               [4.11, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    delta_v_table = []
    rho_air = 1.163
    delta_rho_air = 0.001

    for row in range(len(v_table_1)):
        v_table_1[row] = np.array(v_table_1[row])
        P_d = 9.81
        g = 9.81
        delta_h = 1
        h = 11
        delta_P_d = np.sqrt(2*((rho_air*g*delta_h)**2 + (delta_rho_air*g*h)))
        delta_v = np.sqrt((delta_P_d/(2*np.sqrt(2*P_d/rho_air)))**2 + (-1*delta_rho_air*np.sqrt(2*P_d)/(2*rho_air**1.5))**2)
        delta_v_table.append(delta_v)
    
    plt.figure(figsize=(25,10))
    plt.title("5 m/s: Air Velocity v [m/s] vs Wake Surface Height [mm]", fontsize = TITLE_SIZE)

    heights = [(-22.5 + 5*i) for i in range(NUM_TUBES)]

    for i in range(len(v_table_1)):
        curve = v_table_1[i]
        plt.plot(heights, curve, marker='o', markersize=7, color=COLORS[i], 
        markeredgecolor="black", markerfacecolor=COLORS[i])
        plt.errorbar(heights, curve, yerr=delta_v_table[i], ls="None", color=COLORS[i], capsize=5)

    alpha = np.array([0, 5, 10, 20, 30])
    
    plt.xlabel('Wake Surface Height [mm]', fontsize = AXIS_SIZE)
    plt.ylabel('Air Velocity v [m/s]', fontsize = AXIS_SIZE)
    plt.legend(['0°', '5°', '10°', '20°', '30°'])
    plt.savefig('./Figure 6.png', bbox_inches='tight')

figure_6_plot()

def figure_7_plot():

    v_table_1 = [[16.93, 16.93, 16.93, 16.93, 15.91, 15.91, 15.91, 16.93, 16.93, 16.93],
               [16.93, 16.93, 16.93, 16.93, 16.93, 16.93, 14.81, 16.93, 16.93, 16.93],
               [16.93, 16.93, 16.93, 16.93, 16.93, 16.93, 12.99, 13.62, 16.93, 16.93],
               [14.23, 8.21, 0, 0, 0, 0, 0, 0, 7.11, 15.91],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    P_d_table = []
    delta_v_table = []
    rho_air = 1.163
    delta_rho_air = 0.001

    for row in v_table_1:
        P_d_table.append(np.array(row)**2*rho_air/2)
    
    print(P_d_table)

    for row in range(len(v_table_1)):
        v_table_1[row] = np.array(v_table_1[row])
        P_d = P_d_table[row]
        g = 9.81
        delta_h = 1
        h = 11
        delta_P_d = np.sqrt(2*((rho_air*g*delta_h)**2 + (delta_rho_air*g*h)))
        delta_v = np.sqrt((delta_P_d/(2*np.sqrt(2*P_d/rho_air)))**2 + (-1*delta_rho_air*np.sqrt(2*P_d)/(2*rho_air**1.5))**2)
        delta_v_table.append(delta_v)
    
    heights = [(-22.5 + 5*i) for i in range(NUM_TUBES)]

    plt.figure(figsize=(25,10))
    plt.title("25 m/s: Air Velocity v [m/s] vs Wake Surface Height [mm]", fontsize = TITLE_SIZE)

    for i in range(len(v_table_1)):
        curve = v_table_1[i]
        plt.plot(heights, curve, marker='o', markersize=7, color=COLORS[i], 
        markeredgecolor="black", markerfacecolor=COLORS[i])
        plt.errorbar(heights, curve, yerr=delta_v_table[i], ls="None", color=COLORS[i], capsize=5)

    alpha = np.array([0, 5, 10, 20, 30])
    
    plt.xlabel('Wake Surface Height [mm]', fontsize = AXIS_SIZE)
    plt.ylabel('Air Velocity v [m/s]', fontsize = AXIS_SIZE)
    plt.legend(['0°', '5°', '10°', '20°', '30°'])
    plt.savefig('./Figure 7.png', bbox_inches='tight')

figure_7_plot()
