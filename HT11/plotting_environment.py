import pandas as pd
import math, csv
import matplotlib.pyplot as plt
import numpy as np
from varname import nameof

###############################################################################
# HELPER FUNCTIONS & CONSTANTS
###############################################################################

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

###############################################################################
# EXERCISE A : THERMOCOUPLE TEMPERATURE VS TIME
###############################################################################


###############################################################################
# EXERCISE A : Calculations
###############################################################################

def exercise_a_calculations():

    xls = pd.read_excel('./Data/HT11C_A.xls')

    # Constants
    V = 12.0
    I = 1.2
    V_err = 0.1
    I_err = 0.1
    L_hot = 0.03
    VALUES["L_hot"] = L_hot
    D = 0.025
    A = math.pi * (D ** 2) / 4
    VALUES["A"] = A
    steady_state = 116

    # q_elec
    q_elec = V * I
    VALUES["q_elec"] = q_elec
    q_elec_err = math.sqrt((V_err * I)**2 + (V * I_err)**2)
    VALUES["q_elec_err"] = q_elec_err

    for i in range(1, 9):
        VALUES[f"T_{i}"] = list(xls[f"Temp T{i} [deg C]"])[steady_state - 1]

    VALUES["V"] = V
    VALUES["I"] = I
    VALUES["Flow Rate"] = 1.5

    # k_brass
    T_1 = list(xls["Temp T1 [deg C]"])[steady_state - 1]
    T_3 = list(xls["Temp T3 [deg C]"])[steady_state - 1]
    T_diff = T_1 - T_3
    T_diff_err = math.sqrt(temp_err(T_1) + temp_err(T_3))
    k_brass = q_elec * L_hot / (A * T_diff)
    VALUES["k_brass"] = k_brass
    k_brass_err = math.sqrt(((q_elec_err * L_hot)/(A * T_diff))**2 
                            + ((T_diff_err * q_elec * L_hot)/(A * T_diff**2))**2)
    VALUES["k_brass_err"] = k_brass_err

    # q_int
    T_4 = list(xls["Temp T4 [deg C]"])[steady_state - 1]
    T_5 = list(xls["Temp T5 [deg C]"])[steady_state - 1]
    T_diff = T_4 - T_5
    T_diff_err = math.sqrt(temp_err(T_4) + temp_err(T_5))
    L_int = 0.015
    VALUES["L_int"] = L_int
    q_int = (k_brass * A * T_diff) / L_int
    VALUES["q_int"] = q_int
    q_int_err = math.sqrt((A * T_diff * k_brass_err / L_int)**2 + (A * T_diff_err * k_brass / L_int)**2)
    VALUES["q_int_err"] = q_int_err

    # q_cool
    T_6 = list(xls["Temp T6 [deg C]"])[steady_state - 1]
    T_8 = list(xls["Temp T8 [deg C]"])[steady_state - 1]
    T_diff = T_6 - T_8
    T_diff_err = math.sqrt(temp_err(T_6) + temp_err(T_8))
    L_cool = 0.03
    VALUES["L_cool"] = L_cool
    q_cool = (k_brass * A * T_diff) / L_cool
    VALUES["q_cool"] = q_cool
    q_cool_err = math.sqrt((A * T_diff * k_brass_err / L_cool)**2 + (A * T_diff_err * k_brass / L_cool)**2)
    VALUES["q_cool_err"] = q_cool_err

    # contact resistances
    L_34 = 0.0075 * 2
    L_56 = 0.0075 * 2
    L_45 = 0.015
    VALUES["L_34"] = L_34
    VALUES["L_56"] = L_56
    VALUES["L_45"] = L_45
    R_brass_34 = L_34 / (k_brass * A)
    R_c_hot = (T_3 - T_4) / q_elec - R_brass_34
    VALUES["R_c_hot"] = R_c_hot

    R_brass_56 = L_56 / (k_brass * A)
    R_c_cold = (T_5 - T_6) / q_elec - R_brass_56
    VALUES["R_c_cold"] = R_c_cold

    R_int = L_45 / (k_brass * A)
    VALUES["R_int"] = R_int

###############################################################################
# EXERCISE A : Figure 1 Plot
###############################################################################

def figure_1_plot():

    xls = pd.read_excel('./Data/HT11C_A.xls')
    
    # Line Plotting environment
    plt.figure(figsize=(25,10))
    plt.title("Thermocouple Temperature T8 (C) vs Time (s)", fontsize = TITLE_SIZE)

    # Line plots
    times = list(map(sample_to_time, xls["Sample Number"]))
    temps = list(xls["Temp T8 [deg C]"])
    temps_err = list(map(temp_err, temps))
    plt.xlim([0, max(times)])
    plt.ylim(min(temps) - 2, max(temps) + 2)
    plt.plot(times,temps)

    # Steady state plots
    steady_state = 116
    step_size = 5
    plt.plot(sample_to_time(steady_state), temps[steady_state - 1], 
        marker="o", markersize=10, color=COLORS[0], 
        markeredgecolor="black", markerfacecolor="black")
    plt.errorbar(times[::step_size], temps[::step_size], yerr=temps_err[::step_size], ls="None", color=COLORS[0], capsize=5)

    plt.xlabel('Time (s)', fontsize = AXIS_SIZE)
    plt.ylabel('Thermocouple Temperature (C)', fontsize = AXIS_SIZE)
    plt.savefig('./Figure 1.png', bbox_inches='tight')
    # plt.show() 

figure_1_plot()

###############################################################################
# EXERCISE A : Figure 2 Plot
###############################################################################

def figure_2_plot():

    xls = pd.read_excel('./Data/HT11C_A.xls')

    # Line Plotting environment
    plt.figure(figsize=(25,10))
    plt.title("Thermocouple Temperature Distribution", fontsize = TITLE_SIZE)

    # Line plots
    steady_state = 116
    labels = [f"T{i}" for i in range(1, 9)]
    temps = list(map(lambda s: list(xls[f"Temp {s} [deg C]"])[steady_state], labels))
    temps_err = list(map(temp_err, temps))

    plt.errorbar(labels, temps, yerr=temps_err, ls="None", color=COLORS[0], capsize=5)

    plt.plot(labels, temps, 
        marker="o", markersize=7, color=COLORS[0], 
        markeredgecolor="black", markerfacecolor=COLORS[0])

    plt.xlabel('Thermocouple No.', fontsize = AXIS_SIZE)
    plt.ylabel('Thermocouple Temperature (C)', fontsize = AXIS_SIZE)
    plt.savefig('./Figure 2.png', bbox_inches='tight')
    # plt.show() 

figure_2_plot()


###############################################################################
# EXERCISE B : Calculations
###############################################################################

def exercise_b_calculations():

    xls = pd.read_excel('./Data/HT11C_B.xls')

    # Constants
    V = 12.0
    I = 1.2
    V_err = 0.1
    I_err = 0.1
    L_hot = 0.03
    VALUES["L_hot"] = L_hot
    D = 0.025
    A = math.pi * (D ** 2) / 4
    VALUES["A"] = A
    steady_state = 46

    # q_elec
    q_elec = V * I
    VALUES["q_elec"] = q_elec
    q_elec_err = math.sqrt((V_err * I)**2 + (V * I_err)**2)
    VALUES["q_elec_err"] = q_elec_err

    for i in range(1, 9):
        VALUES[f"T_{i}"] = list(xls[f"Temp T{i} [deg C]"])[steady_state - 1]

    VALUES["V"] = V
    VALUES["I"] = I
    VALUES["Flow Rate"] = 1.35

    # k_brass
    k_brass = 151.415
    VALUES["k_brass"] = k_brass

    # q_int
    T_3 = list(xls["Temp T3 [deg C]"])[steady_state - 1]
    T_4 = list(xls["Temp T4 [deg C]"])[steady_state - 1]
    T_5 = list(xls["Temp T5 [deg C]"])[steady_state - 1]
    T_6 = list(xls["Temp T6 [deg C]"])[steady_state - 1]
    T_8 = list(xls["Temp T8 [deg C]"])[steady_state - 1]

    # contact resistances
    L_34 = 0.0075 * 2
    L_56 = 0.0075 * 2
    L_45 = 0.015
    VALUES["L_34"] = L_34
    VALUES["L_56"] = L_56
    VALUES["L_45"] = L_45
    R_brass_34 = L_34 / (k_brass * A)
    R_c_hot = (T_3 - T_4) / q_elec - R_brass_34
    VALUES["R_c_hot"] = R_c_hot

    R_brass_56 = L_56 / (k_brass * A)
    R_c_cold = (T_5 - T_6) / q_elec - R_brass_56
    VALUES["R_c_cold"] = R_c_cold

    dT_hot = R_c_hot * q_elec
    dT_cold = R_c_cold * q_elec
    VALUES["dT_cold"] = dT_cold
    VALUES["dT_hot"] = dT_hot

###############################################################################
# EXERCISE B : THERMOCOUPLE TEMPERATURE VS TIME
###############################################################################


def figure_3_plot():

    xls = pd.read_excel('./Data/HT11C_B.xls')

    # Line Plotting environment
    plt.figure(figsize=(25,10))
    plt.title("Thermocouple Temperature T8 (C) vs Time (s)", fontsize = TITLE_SIZE)

    # Line plots
    times = list(map(sample_to_time, xls["Sample Number"]))
    temps = list(xls["Temp T8 [deg C]"])
    temps_err = list(map(temp_err, temps))
    plt.xlim([0, max(times)])
    plt.ylim(min(temps) - 2, max(temps) + 2)
    plt.plot(times,temps)

    # Steady state plots
    steady_state = 46
    step_size = 5
    plt.plot(sample_to_time(steady_state), temps[steady_state - 1], 
        marker="o", markersize=10, color=COLORS[0], 
        markeredgecolor="black", markerfacecolor="black")
    plt.errorbar(times[::step_size], temps[::step_size], yerr=temps_err[::step_size], ls="None", color=COLORS[0], capsize=5)

    plt.xlabel('Time (s)', fontsize = AXIS_SIZE)
    plt.ylabel('Thermocouple Temperature (C)', fontsize = AXIS_SIZE)
    plt.savefig('./Figure 3.png', bbox_inches='tight')
    # plt.show() 

figure_3_plot()

def figure_4_plot():

    xls = pd.read_excel('./Data/HT11C_B.xls')

    # Line Plotting environment
    plt.figure(figsize=(25,10))
    plt.title("Thermocouple Temperature Distribution", fontsize = TITLE_SIZE)

    # Line plots
    steady_state = 46
    labels = [f"T{i}" for i in range(1, 9)]
    temps = list(map(lambda s: list(xls[f"Temp {s} [deg C]"])[steady_state], labels))
    temps_err = list(map(temp_err, temps))

    plt.errorbar(labels, temps, yerr=temps_err, ls="None", color=COLORS[0], capsize=5)

    plt.plot(labels, temps, 
        marker="o", markersize=7, color=COLORS[0], 
        markeredgecolor="black", markerfacecolor=COLORS[0])

    steady_state = 116
    labels = [f"T{i}" for i in range(1, 9)]
    temps = list(map(lambda s: list(pd.read_excel('./Data/HT11C_A.xls')[f"Temp {s} [deg C]"])[steady_state], labels))
    temps_err = list(map(temp_err, temps))

    plt.errorbar(labels, temps, yerr=temps_err, ls="None", color=COLORS[1], capsize=5)

    plt.plot(labels, temps, 
        marker="o", markersize=7, color=COLORS[1], 
        markeredgecolor="black", markerfacecolor=COLORS[1])

    plt.xlabel('Thermocouple No.', fontsize = AXIS_SIZE)
    plt.ylabel('Thermocouple Temperature (C)', fontsize = AXIS_SIZE)
    plt.legend(["Exercise B", "Exercise A"])
    plt.savefig('./Figure 4.png', bbox_inches='tight')
    # plt.show() 

figure_4_plot()

###############################################################################
# EXERCISE C : Calculations
###############################################################################

def exercise_c_calculations():
    # Data Parsing
    xls = pd.read_excel('./Data/HT11C_C.xls')

    # Constants
    V = 12.0
    I = 1.2
    V_err = 0.1
    I_err = 0.1
    L_hot = 0.03
    VALUES["L_hot"] = L_hot
    D = 0.013
    A = math.pi * (D ** 2) / 4
    VALUES["A"] = A
    steady_state = 65

    # q_elec
    q_elec = V * I
    VALUES["q_elec"] = q_elec
    q_elec_err = math.sqrt((V_err * I)**2 + (V * I_err)**2)
    VALUES["q_elec_err"] = q_elec_err

    for i in range(1, 9):
        VALUES[f"T_{i}"] = list(xls[f"Temp T{i} [deg C]"])[steady_state - 1]

    VALUES["V"] = V
    VALUES["I"] = I
    VALUES["Flow Rate"] = 1.35

    # k_brass
    k_brass = 151.415
    VALUES["k_brass"] = k_brass

    # q_sample
    T_hot = list(xls["Temp T3 [deg C]"])[steady_state - 1]
    T_cold = list(xls["Temp T6 [deg C]"])[steady_state - 1]
    T_diff = T_hot - T_cold
    VALUES["T_hot"] = T_hot
    VALUES["T_cold"] = T_cold
    L_sample = 0.03
    VALUES["L_sample"] = L_sample
    q_sample = (k_brass * A * T_diff) / L_sample
    VALUES["q_sample"] = q_sample

###############################################################################
# EXERCISE C : THERMOCOUPLE TEMPERATURE VS TIME
###############################################################################

def figure_5_plot():
    # Data Parsing
    xls = pd.read_excel('./Data/HT11C_C.xls')
    
    # Line Plotting environment
    plt.figure(figsize=(25,10))
    plt.title("Thermocouple Temperature T8 (C) vs Time (s)", fontsize = TITLE_SIZE)

    # Line plots
    times = list(map(sample_to_time, xls["Sample Number"]))
    temps = list(xls["Temp T8 [deg C]"])
    temps_err = list(map(temp_err, temps))
    plt.xlim([0, max(times)])
    plt.ylim(min(temps) - 2, max(temps) + 2)
    plt.plot(times,temps)

    # Steady state plots
    steady_state = 65
    step_size = 5
    plt.plot(sample_to_time(steady_state), temps[steady_state - 1], 
        marker="o", markersize=10, color=COLORS[0], 
        markeredgecolor="black", markerfacecolor="black")
    plt.errorbar(times[::step_size], temps[::step_size], yerr=temps_err[::step_size], ls="None", color=COLORS[0], capsize=5)

    plt.xlabel('Time (s)', fontsize = AXIS_SIZE)
    plt.ylabel('Thermocouple Temperature (C)', fontsize = AXIS_SIZE)
    plt.savefig('./Figure 5.png', bbox_inches='tight')
    # plt.show() 

figure_5_plot()

def figure_6_plot():
    # Data Parsing
    xls = pd.read_excel('./Data/HT11C_C.xls')
    
    # Line Plotting environment
    plt.figure(figsize=(25,10))
    plt.title("Thermocouple Temperature Distribution", fontsize = TITLE_SIZE)

    # Line plots
    steady_state = 65
    labels = [f"T{i}" for i in range(1, 9)]
    temps = list(map(lambda s: list(xls[f"Temp {s} [deg C]"])[steady_state], labels))
    temps_err = list(map(temp_err, temps))

    plt.errorbar(labels, temps, yerr=temps_err, ls="None", color=COLORS[0], capsize=5)

    plt.plot(labels, temps, 
        marker="o", markersize=7, color=COLORS[0], 
        markeredgecolor="black", markerfacecolor=COLORS[0])

    steady_state = 116
    labels = [f"T{i}" for i in range(1, 9)]
    temps = list(map(lambda s: list(pd.read_excel('./Data/HT11C_A.xls')[f"Temp {s} [deg C]"])[steady_state], labels))
    temps_err = list(map(temp_err, temps))

    plt.errorbar(labels, temps, yerr=temps_err, ls="None", color=COLORS[1], capsize=5)

    plt.plot(labels, temps, 
        marker="o", markersize=7, color=COLORS[1], 
        markeredgecolor="black", markerfacecolor=COLORS[1])

    plt.xlabel('Thermocouple No.', fontsize = AXIS_SIZE)
    plt.ylabel('Thermocouple Temperature (C)', fontsize = AXIS_SIZE)
    plt.legend(["Exercise C", "Exercise A"])
    plt.savefig('./Figure 6.png', bbox_inches='tight')
    # plt.show() 

figure_6_plot()

###############################################################################
# EXERCISE D : Calculations
###############################################################################

def exercise_d_calculations():
    # Data Parsing
    xls = pd.read_excel('./Data/HT11C_D.xls')
    
    # Constants
    V = 1.6
    I = 0.2
    V_err = 0.1
    I_err = 0.1
    L_hot = 0.03
    VALUES["L_hot"] = L_hot
    D = 0.025
    A = math.pi * (D ** 2) / 4
    VALUES["A"] = A
    steady_state = 46

    # q_elec
    q_elec = V * I
    VALUES["q_elec"] = q_elec
    q_elec_err = math.sqrt((V_err * I)**2 + (V * I_err)**2)
    VALUES["q_elec_err"] = q_elec_err

    for i in range(1, 9):
        VALUES[f"T_{i}"] = list(xls[f"Temp T{i} [deg C]"])[steady_state - 1]

    VALUES["V"] = V
    VALUES["I"] = I
    VALUES["Flow Rate"] = 1.35

    # k_paper
    T_3 = list(xls["Temp T3 [deg C]"])[steady_state - 1]
    T_6 = list(xls["Temp T6 [deg C]"])[steady_state - 1]
    T_diff = T_3 - T_6
    T_diff_err = math.sqrt(temp_err(T_3) + temp_err(T_6))
    k_paper = q_elec * L_hot / (A * T_diff)
    VALUES["k_paper"] = k_paper
    k_paper_err = math.sqrt(((q_elec_err * L_hot)/(A * T_diff))**2 
                            + ((T_diff_err * q_elec * L_hot)/(A * T_diff**2))**2)
    VALUES["k_paper_err"] = k_paper_err
    
    x = [i * 0.015 for i in range(3)]
    T5_reg = [list(xls[f"Temp T{i} [deg C]"])[steady_state - 1] for i in range(6, 9)]
    T4_reg = [list(xls[f"Temp T{i} [deg C]"])[steady_state - 1] for i in range(1, 4)]
    (m, b) = np.polyfit(x, T5_reg, 1)
    T_5 = m * -0.015 + b
    (m, b) = np.polyfit(x, T4_reg, 1)
    T_4 = m * -0.015 + b
    VALUES["T_5_est"] = T_5
    VALUES["T_4_est"] = T_4

###############################################################################
# EXERCISE D : THERMOCOUPLE TEMPERATURE VS TIME
###############################################################################

def figure_7_plot():
    # Data Parsing
    xls = pd.read_excel('./Data/HT11C_D.xls')

    # Line Plotting environment
    plt.figure(figsize=(25,10))
    plt.title("Thermocouple Temperature T8 (C) vs Time (s)", fontsize = TITLE_SIZE)

    # Line plots
    times = list(map(sample_to_time, xls["Sample Number"]))
    temps = list(xls["Temp T8 [deg C]"])
    temps_err = list(map(temp_err, temps))
    plt.xlim([0, max(times)])
    plt.ylim(min(temps) - 2, max(temps) + 2)
    plt.plot(times,temps)

    # Steady state plots
    steady_state = 195
    step_size = 5
    plt.plot(sample_to_time(steady_state), temps[steady_state - 1], 
        marker="o", markersize=10, color=COLORS[0], 
        markeredgecolor="black", markerfacecolor="black")
    plt.errorbar(times[::step_size], temps[::step_size], yerr=temps_err[::step_size], ls="None", color=COLORS[0], capsize=5)

    plt.xlabel('Time (s)', fontsize = AXIS_SIZE)
    plt.ylabel('Thermocouple Temperature (C)', fontsize = AXIS_SIZE)
    plt.savefig('./Figure 7.png', bbox_inches='tight')
    # plt.show() 

figure_7_plot()

print("Saved all figures.")

def calculations():
    exercise_a_calculations()
    print_values("A")
    VALUES.clear()
    exercise_b_calculations()
    print_values("B")
    VALUES.clear()
    exercise_c_calculations()
    print_values("C")
    VALUES.clear()
    exercise_d_calculations()
    print_values("D")

calculations()