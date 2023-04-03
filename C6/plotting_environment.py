import pandas as pd
import math, csv
import matplotlib.pyplot as plt
import numpy as np

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

TITLE_SIZE = 15
AXIS_SIZE = 13

###############################################################################
# EXERCISE A : Calculations
###############################################################################

def exercise_a_calculations():
    xls = pd.read_excel('./Data/Table 1, 2.xlsx')

    t1_columns = ['d [mm]', 'Q [m^3/s]', 'T [C]', 'h1 [m]', 'h2 [m]', 
                  'V_avg [m/s]', 'v [m^2/s]', 'Re_D', 'f_e', 'f_c', 'h_e [m]', 
                  'h_c [m]']

    for column_name in t1_columns:
        get_average_save(xls, column_name)
    print_values('1')
    VALUES.clear()

    t2_columns = ['d_err [mm]', 'Q_err [m^3/s]', 'T_err [C]', 'h1_err [m]', 'h2_err [m]', 
                  'V_avg_err [m/s]', 'Re_D_err', 'f_e_err', 'f_c_err', 'h_e_err [m]', 
                  'h_c_err [m]']

    for column_name in t2_columns:
        get_average_save(xls, column_name)
    print_values('2')
    
def exercise_a_save_values():

    xls = pd.read_excel('./Data/Table 1, 2.xlsx')

    t1_columns = ['d [mm]', 'Q [m^3/s]', 'T [C]', 'h1 [m]', 'h2 [m]', 
                  'V_avg [m/s]', 'v [m^2/s]', 'Re_D', 'f_e', 'f_c', 'h_e [m]', 
                  'h_c [m]', 'd_err [mm]', 'Q_err [m^3/s]', 'T_err [C]', 
                  'h1_err [m]', 'h2_err [m]', 'V_avg_err [m/s]', 'Re_D_err', 
                  'f_e_err', 'f_c_err', 'h_e_err [m]', 'h_c_err [m]']

    for column_name in t1_columns:
        get_average_save(xls, column_name)

def figure_1_plot():

    exercise_a_save_values()

    d = VALUES['d [mm]']
    Q = VALUES['Q [m^3/s]']
    h_e = VALUES['h_e [m]']
    h_c = VALUES['h_c [m]']
    delta_h_e = VALUES['h_e_err [m]']
    delta_h_c = VALUES['h_c_err [m]']


    plt.figure(figsize=(25,10))
    plt.title("Measured Head Loss h_e [m] vs. Flow Rate Q [m^3/s]", fontsize = TITLE_SIZE)

    curves_Q = []
    curves_h_e = []
    curves_h_c = []
    curves_delta_h_e = []
    curves_delta_h_c = []
    seen = set()
    for i in range(len(d)):
        if d[i] in seen:
            curves_Q[-1].append(Q[i])
            curves_h_e[-1].append(h_e[i])
            curves_delta_h_e[-1].append(delta_h_e[i])
            curves_h_c[-1].append(h_c[i])
            curves_delta_h_c[-1].append(delta_h_c[i])
        else:
            curves_Q.append([])
            curves_h_e.append([])
            curves_delta_h_e.append([])
            curves_h_c.append([])
            curves_delta_h_c.append([])
            seen.add(d[i])

    for plot in range(len(curves_Q) * 2):
        i = plot // 2
        if plot % 2 == 0:
            plt.plot(curves_Q[i], curves_h_e[i], marker='o', markersize=7, 
                        color=COLORS[plot],  markeredgecolor="black", 
                        markerfacecolor=COLORS[plot])
            plt.errorbar(curves_Q[i], curves_h_e[i], yerr=curves_delta_h_e[i], 
                        ls="None", color=COLORS[plot], capsize=5)
        else:
            plt.plot(curves_Q[i], curves_h_c[i], marker='o', markersize=7, 
                        color=COLORS[plot],  markeredgecolor="black", 
                        markerfacecolor=COLORS[plot])
            plt.errorbar(curves_Q[i], curves_h_c[i], yerr=curves_delta_h_c[i], 
                        ls="None", color=COLORS[plot], capsize=5)

    plt.xlabel('Flow Rate Q [m^3/s]', fontsize = AXIS_SIZE)
    plt.ylabel('Calculated & Measured Head Loss h_c, h_e [m]', fontsize = AXIS_SIZE)
    plt.legend(['Measured Tube 10', 'Calculated Tube 10', 'Measured Tube 9', 'Calculated Tube 9',
                'Measured Tube 8', 'Calculated Tube 8', 'Measured Tube 7', 'Calculated Tube 7'])
    plt.savefig('./Figure 1.png', bbox_inches='tight')

def figure_2_plot():

    exercise_a_save_values()

    d = VALUES['d [mm]']
    Re = VALUES['Re_D']
    f_e = VALUES['f_e']
    f_c = VALUES['f_c']
    delta_f_e = VALUES['f_e_err']
    delta_f_c = VALUES['f_c_err']

    plt.figure(figsize=(25,10))
    plt.title("Measured Friction Coefficient f_e vs. Reynolds Number Re_D", fontsize = TITLE_SIZE)

    curves_Re = []
    curves_f_e = []
    curves_f_c = []
    curves_delta_f_e = []
    curves_delta_f_c = []
    seen = set()
    for i in range(len(d)):
        if d[i] in seen:
            curves_Re[-1].append(Re[i])
            curves_f_e[-1].append(f_e[i])
            curves_delta_f_e[-1].append(delta_f_e[i])
            curves_f_c[-1].append(f_c[i])
            curves_delta_f_c[-1].append(delta_f_c[i])
        else:
            curves_Re.append([])
            curves_f_e.append([])
            curves_delta_f_e.append([])
            curves_f_c.append([])
            curves_delta_f_c.append([])
            seen.add(d[i])

    for plot in range(len(curves_Re) * 2):
        i = plot // 2
        if plot % 2 == 0:
            plt.plot(curves_Re[i], curves_f_e[i], marker='o', markersize=7, 
                        color=COLORS[plot],  markeredgecolor="black", 
                        markerfacecolor=COLORS[plot])
            plt.errorbar(curves_Re[i], curves_f_e[i], yerr=curves_delta_f_e[i], 
                        ls="None", color=COLORS[plot], capsize=5)
        else:
            plt.plot(curves_Re[i], curves_f_c[i], marker='o', markersize=7, 
                        color=COLORS[plot],  markeredgecolor="black", 
                        markerfacecolor=COLORS[plot])
            plt.errorbar(curves_Re[i], curves_f_c[i], yerr=curves_delta_f_c[i], 
                        ls="None", color=COLORS[plot], capsize=5)
    VALUES.clear()

    plt.xlabel('Reynolds Number', fontsize = AXIS_SIZE)
    plt.ylabel('Measured Friction Coefficient f_e', fontsize = AXIS_SIZE)
    plt.legend(['Measured Tube 10', 'Calculated Tube 10', 'Measured Tube 9', 'Calculated Tube 9',
                'Measured Tube 8', 'Calculated Tube 8', 'Measured Tube 7', 'Calculated Tube 7'])
    plt.savefig('./Figure 2.png', bbox_inches='tight')

def figure_3_plot():

    exercise_a_save_values()

    d = VALUES['d [mm]']
    V = VALUES['V_avg [m/s]']
    h_e = VALUES['h_e [m]']
    h_c = VALUES['h_c [m]']
    delta_h_e = VALUES['h_e_err [m]']
    delta_h_c = VALUES['h_c_err [m]']


    plt.figure(figsize=(25,10))
    plt.title("Measured Head Loss h_e [m] vs. Flow Velocity V [m/s]", fontsize = TITLE_SIZE)

    curves_V = []
    curves_h_e = []
    curves_delta_h_e = []
    seen = set()
    for i in range(len(d)):
        if d[i] in seen:
            curves_V[-1].append(V[i])
            curves_h_e[-1].append(h_e[i])
            curves_delta_h_e[-1].append(delta_h_e[i])
        else:
            curves_V.append([])
            curves_h_e.append([])
            curves_delta_h_e.append([])
            seen.add(d[i])

    for i in range(len(curves_V)):
        a, b, c = np.polyfit(curves_V[i], curves_h_e[i], 2)
        x = np.array(curves_V[i])
        plt.plot(x, curves_h_e[i], marker='o', markersize=7, 
                    color=COLORS[i],  markeredgecolor="black", 
                    markerfacecolor=COLORS[i])
        plt.plot(x, a*(x**2)+b*x+c, color='black')
        plt.errorbar(x, curves_h_e[i], yerr=curves_delta_h_e[i], 
                    ls="None", color=COLORS[i], capsize=5)

    plt.xlabel('Flow Velocity V [m/s]', fontsize = AXIS_SIZE)
    plt.ylabel('Measured Head Loss h_e [m]', fontsize = AXIS_SIZE)
    plt.legend(['Tube 10', 'Tube 10 Fit', 'Tube 9', 'Tube 9 Fit',
                'Tube 8', 'Tube 8 Fit', 'Tube 7', 'Tube 7 Fit'])
    plt.savefig('./Figure 3.png', bbox_inches='tight')

def exercise_b_calculations():
    xls = pd.read_excel('./Data/Table 3, 4.xlsx')

    t1_columns = ['d [mm]', 'Q [m^3/s]', 'T [C]', 'V_avg [m/s]', 'v [m^2/s]',
                  'Re_D', 'h1 [m]', 'h2 [m]', 'h_K [m]', 'K', 'L_eq [m]']

    VALUES.clear()
    for column_name in t1_columns:
        get_average_save(xls, column_name)
    print_values('3')
    VALUES.clear()

    accessories = list(xls['Fitting  Type'])
    K = list(xls['K'])
    rows = len(accessories)
    accessories_avg = [accessories[row] for row in range(0, rows, 5)]
    VALUES['Accessory'] = accessories_avg

    get_average_save(xls, 'Q [m^3/s]')

    averages = []
    stdevs = []
    for row in range(rows // 5):
        buckets = [K[row * 5 + i] for i in range(5)]
        averages.append(np.average(buckets))
        stdevs.append(np.std(buckets))
    VALUES['K_avg'] = averages
    VALUES['K_stdev'] = stdevs
    print_values('4')
    VALUES.clear()

def exercise_d_calculations():
    xls = pd.read_excel('./Data/Table 5.xlsx')

    d_O = 21/1000
    d_V = 14.5/1000
    d_1 = 24/1000
    delta_d = 0.0005
    A_O = math.pi / 4 * (21 / 1000)**2
    A_V = math.pi / 4 * (14.5 / 1000)**2
    A_1 = math.pi / 4 * (24 / 1000)**2
    get_O = lambda s: np.array(list(map(lambda x: xls[s][x], 
            filter(lambda x: x % 10 < 5, [i for i in range(len(xls[s]))]))))
    get_V = lambda s: np.array(list(map(lambda x: xls[s][x], 
            filter(lambda x: x % 10 >= 5, [i for i in range(len(xls[s]))]))))
    
    # Orifice Table
    VALUES['Q_m [l/s]'] = Q_meas_O = get_O('Q [l/s]')
    VALUES['V_m [m/s]'] = get_O('V [m/s]')
    VALUES['T [C]'] = get_O('T [C]')
    VALUES['v [m^2/s]'] = get_O('v [m^2/s]')
    VALUES['Re'] = get_O('Re')
    VALUES['H_o1'] = get_O('h_1 [m]')
    VALUES['H_o2'] = get_O('h_2 [m]')
    delta_h = VALUES['H_o1'] - VALUES['H_o2']
    VALUES['Q_O'] = 0.62 * A_O * np.sqrt(2 * 9.81 * delta_h) / (1 - (A_O/A_1)**2)*1000
    VALUES['Q_MO'] = (Q_meas_O - VALUES['Q_O'])/VALUES['Q_O']*100
    t1 = 8.85 * d_O**5 * d_1**2 * np.sqrt(delta_h) / (d_1**4 - d_O**4)**1.5
    t2 = 2 * d_O * np.sqrt(19.6 * delta_h * d_1**4 / (d_1**4 - d_O**4))
    Q_O = 0.62 * (np.pi / 4) * delta_d * (t1 + t2)
    t1 = -8.85 * d_O**6 * d_1 * np.sqrt(delta_h) / (d_1**4 - d_O**4)**1.5
    Q_t =  0.62 * (np.pi / 4) * delta_d * t1
    VALUES['delta_Q_O'] = np.sqrt(Q_O**2 + Q_t**2)

    for name in VALUES:
        VALUES[name] = get_average_save_0(VALUES[name])

    print_values('5.1')
    VALUES.clear()

    # Venturi Table
    VALUES['Q_m [l/s]'] = Q_meas_O = get_V('Q [l/s]')
    VALUES['V_m [m/s]'] = get_V('V [m/s]')
    VALUES['T [C]'] = get_V('T [C]')
    VALUES['v [m^2/s]'] = get_V('v [m^2/s]')
    VALUES['Re'] = get_V('Re')
    Q_meas_V = get_V('Q [l/s]')
    VALUES['H_v1'] = get_V('h_1 [m]')
    VALUES['H_v2'] = get_V('h_2 [m]')
    dH_V = VALUES['H_v1'] - VALUES['H_v2']
    VALUES['Q_V'] = 0.98 * A_V * np.sqrt(2 * 9.81 * dH_V) / (1 - (A_V/A_1)**2)*1000
    VALUES['Q_MV'] = (Q_meas_V - VALUES['Q_V'])/VALUES['Q_V']*100
    t1 = 8.85 * d_V**5 * d_1**2 * np.sqrt(delta_h) / (d_1**4 - d_V**4)**1.5
    t2 = 2 * d_V * np.sqrt(19.6 * delta_h * d_1**4 / (d_1**4 - d_V**4))
    Q_V = 0.62 * (np.pi / 4) * delta_d * (t1 + t2)
    t1 = -8.85 * d_V**6 * d_1 * np.sqrt(delta_h) / (d_1**4 - d_V**4)**1.5
    Q_t =  0.62 * (np.pi / 4) * delta_d * t1
    VALUES['delta_Q_V'] = np.sqrt(Q_V**2 + Q_t**2)

    for name in VALUES:
        VALUES[name] = get_average_save_0(VALUES[name])
        
    print_values('5.2')
    VALUES.clear()

def calculations():
    exercise_a_calculations()
    figure_1_plot()
    figure_2_plot()
    figure_3_plot()
    exercise_b_calculations()
    exercise_d_calculations()

calculations()

print("Saved all figures.")

