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
    return y1 + (y2 - y1) * (x3 - x1) / (x2 - x1)


def find_steady_state(data : list[float]) ->int:
    assert(len(data) > 10)
    init = (data[13] - data[3]) / 10
    for i in range(0, len(data) - 10):
        slope = (data[i+10] - data[i]) / 10
        if abs(slope) < abs(init) * 0.1:
            if i < int(0.7 * len(data)):
                continue
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

    xls = pd.read_excel('./Data/C5_HTFin.xlsx')

    scan_start = 400
    scan_end = 2000
    scans = [i+1 for i in range(scan_end-scan_start)]
    rod_materials = ['Brass', 'Copper', 'Steel', 'Aluminum']
    rods = [[list(xls[f'{rod}-{i+1}'])[scan_start:scan_end] for i in range(5)] for rod in rod_materials]
    figures = [(0, 0), (0, 1), (1, 0), (1, 1)]

    plt.figure(figsize=(25,25))
    figure, axis = plt.subplots(2, 2, figsize=(10, 10))

    # free
    for i in range(len(rods)):
        if i >= 2:
            scan_start = 2347
            scan_end = 3500
            scans = [i+1 for i in range(scan_end-scan_start)]
            rods = [[list(xls[f'{rod}-{i+1}'])[scan_start:scan_end] for i in range(5)] for rod in rod_materials]
        rod_data = rods[i]
        rod = rod_materials[i]
        rod_ss = [1124, 1243, 825, 921]

        (p1, p2) = figures[i]
        axis[p1, p2].set_title(f"{rod} Temperature vs. Scan Number")
        axis[p1, p2].set(xlabel='Scan Number', ylabel='Temperature [C]')
        for thermocouple in range(5):
            axis[p1, p2].plot(scans, rod_data[thermocouple], color=COLORS[thermocouple],
                        markerfacecolor=COLORS[thermocouple])
            axis[p1, p2].errorbar(scans[::ERROR_INT], rod_data[thermocouple][::ERROR_INT], yerr=0.5, 
                        ls="None", color=COLORS[thermocouple], capsize=5)
        # axis[p1, p2].xlabel(, fontsize = AXIS_SIZE)
        # axis[p1, p2].ylabel(, fontsize = AXIS_SIZE)
        axis[p1, p2].legend([f'{rod}-{i+1}' for i in range(5)])
    figure.tight_layout()
    plt.savefig(f'./Figure 1.png', bbox_inches='tight')

def figure_2_plot():

    xls = pd.read_excel('./Data/C5_HTFin.xlsx')

    scan_start = 2347
    scan_end = 3500
    scans = [i+1 for i in range(scan_end-scan_start)]
    rod_materials = ['Brass', 'Copper', 'Steel', 'Aluminum']
    rods = [[list(xls[f'{rod}-{i+1}'])[scan_start:scan_end] for i in range(5)] for rod in rod_materials]
    figures = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    plt.figure(figsize=(25,25))
    figure, axis = plt.subplots(2, 2, figsize=(10, 10))

    # forced
    for i in range(len(rods)):
        if i >= 2:
            scan_start = 400
            scan_end = 2000
            scans = [i+1 for i in range(scan_end-scan_start)]
            rods = [[list(xls[f'{rod}-{i+1}'])[scan_start:scan_end] for i in range(5)] for rod in rod_materials]
        rod_data = rods[i]
        rod = rod_materials[i]
        rod_ss = [807, 809, 1150, 1126]
        
        (p1, p2) = figures[i]
        axis[p1, p2].set_title(f"{rod} Temperature vs. Scan Number")
        axis[p1, p2].set(xlabel='Scan Number', ylabel='Temperature [C]')
        for thermocouple in range(5):
            axis[p1, p2].plot(scans, rod_data[thermocouple], color=COLORS[thermocouple],
                        markerfacecolor=COLORS[thermocouple])
            axis[p1, p2].errorbar(scans[::ERROR_INT], rod_data[thermocouple][::ERROR_INT], yerr=0.5, 
                        ls="None", color=COLORS[thermocouple], capsize=5)
        # axis[p1, p2].xlabel(, fontsize = AXIS_SIZE)
        # axis[p1, p2].ylabel(, fontsize = AXIS_SIZE)
        axis[p1, p2].legend([f'{rod}-{i+1}' for i in range(5)])
    figure.tight_layout()
    plt.savefig(f'./Figure 2.png', bbox_inches='tight')

def figure_3_plot():
    distances = [0.0254 * 3 * (4-i) for i in range(5)]
    rod_materials = ['Brass', 'Copper', 'Steel', 'Aluminum']
    brass_free = [30.4336, 32.6462, 38.6185, 51.4694, 76.5699]
    copper_free = [46.2017, 50.5833, 53.6397, 60.3475, 68.9532]
    steel_free = [22.2945, 22.5085, 24.3151, 34.2729, 82.7978]
    aluminum_free = [28.5176, 29.329, 32.6868, 39.2744, 51.1504]
    brass_forced = [23.5698, 24.2858, 26.1338, 32.4907, 57.8694]
    copper_forced = [32.4202, 34.7091, 36.3617, 40.8295, 48.889]
    steel_forced = [23.0789, 23.0422, 23.4398, 24.6957, 56.0011]
    aluminum_forced = [25.0694, 25.3451, 26.7059, 29.9053, 40.859]
    T_data = [[30.4336, 32.6462, 38.6185, 51.4694, 76.5699], 
              [46.2017, 50.5833, 53.6397, 60.3475, 68.9532], 
              [22.2945, 22.5085, 24.3151, 34.2729, 82.7978], 
              [28.5176, 29.329, 32.6868, 39.2744, 51.1504], 
              [23.5698, 24.2858, 26.1338, 32.4907, 57.8694], 
              [32.4202, 34.7091, 36.3617, 40.8295, 48.889], 
              [23.0789, 23.0422, 23.4398, 24.6957, 56.0011], 
              [25.0694, 25.3451, 26.7059, 29.9053, 40.859]]

    VALUES['brass_free'] = brass_free
    VALUES['copper_free'] = copper_free
    VALUES['steel_free'] = steel_free
    VALUES['aluminium_free'] = aluminum_free

    VALUES['brass_forced'] = brass_forced
    VALUES['copper_forced'] = copper_forced
    VALUES['steel_forced'] = steel_forced
    VALUES['aluminium_forced'] = aluminum_forced


    print_values('1')

    plt.figure(figsize=(25,10))
    plt.title("Free Convection Temperature [C] vs Fin Position [m]", fontsize = TITLE_SIZE)

    # def f()

    i = 0
    D = 0.0127
    T_inf = 21.7
    L = 0.3048
    k = np.array([110, 401, 15.1, 237])
    h = np.array([23.302330233023305, 18.72187218721872, 20.522052205220522, 37.363736373637366])
    d = np.array([(0.0254) * 3 * (4-i) for i in range(5)])
    # h = [74.86748674867488, 31.463146314631466, 45.06450645064507, 61.80618061806181]
    rods = [brass_free, copper_free, steel_free, aluminum_free]
    for i in range(len(rods)):
        rod = rods[i]
        plt.scatter(distances, rod, marker='o',
                    color=COLORS[i])
        plt.errorbar(distances, rod, yerr=0.5, ls="None", color=COLORS[i], capsize=5)

    for i in range(len(rods)):
        rod = rods[i]
        T_b = T_data[i][-1]
        m = np.sqrt(4*h[i]/(k[i]*D))
        hmk = h[i] / (m * k[i])
        T_est = (T_b - T_inf)*(np.cosh(m*(L-d)) + hmk*np.sinh(m*(L-d)))/(np.cosh(m*L) + hmk*np.sinh(m*L)) + T_inf
        plt.plot(distances, T_est, marker='o', markersize=7, 
                        color=COLORS[i],  markeredgecolor="black", 
                        markerfacecolor=COLORS[i])

    plt.xlabel('Distance from Base [m]', fontsize = AXIS_SIZE)
    plt.ylabel('Temperature [C]', fontsize = AXIS_SIZE)
    plt.legend([f"Experimental {rod} Temperature" for rod in rod_materials]+
               [f"Theoretical {rod} Temperature" for rod in rod_materials])
    plt.savefig('./Figure 3.png', bbox_inches='tight')
    
def figure_4_plot():
    distances = [0.0254 * 3 * (4-i) for i in range(5)]
    rod_materials = ['Brass', 'Copper', 'Steel', 'Aluminum']
    brass_forced = [23.5698, 24.2858, 26.1338, 32.4907, 57.8694]
    copper_forced = [32.4202, 34.7091, 36.3617, 40.8295, 48.889]
    steel_forced = [23.0789, 23.0422, 23.4398, 24.6957, 56.0011]
    aluminum_forced = [25.0694, 25.3451, 26.7059, 29.9053, 40.859]
    T_data = [[23.5698, 24.2858, 26.1338, 32.4907, 57.8694], 
              [32.4202, 34.7091, 36.3617, 40.8295, 48.889], 
              [23.0789, 23.0422, 23.4398, 24.6957, 56.0011], 
              [25.0694, 25.3451, 26.7059, 29.9053, 40.859]]

    VALUES['brass_forced'] = brass_forced
    VALUES['copper_forced'] = copper_forced
    VALUES['steel_forced'] = steel_forced
    VALUES['aluminium_forced'] = aluminum_forced

    VALUES['brass_forced'] = brass_forced
    VALUES['copper_forced'] = copper_forced
    VALUES['steel_forced'] = steel_forced
    VALUES['aluminium_forced'] = aluminum_forced


    print_values('1')

    plt.figure(figsize=(25,10))
    plt.title("Forced Convection Temperature [C] vs Fin Position [m]", fontsize = TITLE_SIZE)

    # def f()

    i = 0
    D = 0.0127
    T_inf = 21.7
    L = 0.3048
    k = np.array([110, 401, 15.1, 237])
    h = np.array([74.86748674867488, 31.463146314631466, 45.06450645064507, 61.80618061806181])
    d = np.array([(0.0254) * 3 * (4-i) for i in range(5)])
    rods = [brass_forced, copper_forced, steel_forced, aluminum_forced]
    for i in range(len(rods)):
        rod = rods[i]
        plt.scatter(distances, rod, marker='o',
                    color=COLORS[i])
        plt.errorbar(distances, rod, yerr=0.5, ls="None", color=COLORS[i], capsize=5)

    for i in range(len(rods)):
        rod = rods[i]
        T_b = T_data[i][-1]
        m = np.sqrt(4*h[i]/(k[i]*D))
        hmk = h[i] / (m * k[i])
        T_est = (T_b - T_inf)*(np.cosh(m*(L-d)) + hmk*np.sinh(m*(L-d)))/(np.cosh(m*L) + hmk*np.sinh(m*L)) + T_inf
        plt.plot(distances, T_est, marker='o', markersize=7, 
                        color=COLORS[i],  markeredgecolor="black", 
                        markerfacecolor=COLORS[i])

    plt.xlabel('Distance from Base [m]', fontsize = AXIS_SIZE)
    plt.ylabel('Temperature [C]', fontsize = AXIS_SIZE)
    plt.legend([f"Experimental {rod} Temperature" for rod in rod_materials]+
               [f"Theoretical {rod} Temperature" for rod in rod_materials])
    plt.savefig('./Figure 4.png', bbox_inches='tight')

def best_fit_parametric_est():
    h_all = np.linspace(0, 200, 10000)
    d = np.array([(0.0254) * 3 * (4-i) for i in range(5)])
    result = []
    D = 0.0127
    T_inf = 21.7
    k = np.array([110, 401, 15.1, 237]*2)
    L = 0.3048
    T_data = [[30.4336, 32.6462, 38.6185, 51.4694, 76.5699], 
              [46.2017, 50.5833, 53.6397, 60.3475, 68.9532], 
              [22.2945, 22.5085, 24.3151, 34.2729, 82.7978], 
              [28.5176, 29.329, 32.6868, 39.2744, 51.1504], 
              [23.5698, 24.2858, 26.1338, 32.4907, 57.8694], 
              [32.4202, 34.7091, 36.3617, 40.8295, 48.889], 
              [23.0789, 23.0422, 23.4398, 24.6957, 56.0011], 
              [25.0694, 25.3451, 26.7059, 29.9053, 40.859]]
    for i in range(8):
        h_min = 0
        F_min = float('inf')
        for h in h_all:
            T_b = T_data[i][-1]
            m = np.sqrt(4*h/(k[i]*D))
            hmk = h / (m * k[i])
            F = 0
            for j in range(len(d)):
                T_est = (T_b - T_inf)*(np.cosh(m*(L-d[j])) + hmk*np.sinh(m*(L-d[j])))/(np.cosh(m*L) + hmk*np.sinh(m*L)) + T_inf
                F += (T_est - T_data[i][j])**2
            if F < F_min:
                h_min = h
                F_min = F
        result.append(h_min)
    return result

print(best_fit_parametric_est())

def table_2_calculations():

    t1_columns = ['T_f [C]', 'U_air [m/s]', 'Re_D', 'Gr_D', 'Pr', 'Nu_D', 
                  'h_conv [W/m^2*K]', 'h_rad [W/m^2*K]', 
                  'h_predicted [W/m^2*K]', 'h_measured [W/m^2*K]']

    VALUES.clear()
    
    T_f = np.array([55.94508, 38.6419])
    T_inf = 21.7
    U_air = np.array([0, 2.972])
    Pr = np.array([0.7029266888, 0.705349134])
    k = np.array([28.45303592*10**-3, 27.1726006*10**-3])
    epsilon = 0.03
    sigma = 5.67*10**-8
    D = 0.0127
    g = 9.81
    beta = 1/(T_inf+273.15)
    v = 15.43165*10**-6
    Re_D = U_air*D/v
    Gr_D = g*beta*(T_f-T_inf)*D**3/v**2
    Ra_D = Gr_D*Pr
    Nu_D_1 = (0.6+0.387*(Ra_D)**(1/6)/(1+(0.559*Pr)**(9/16))**(8/27))**2
    Nu_D_2 = 0.3+(0.62*Re_D**0.5*Pr**(1/3))/(1+(0.4/Pr)**(2/3))**0.25*(1+(Re_D/282000)**(5/8))**0.8
    # Nu_D_2 = 0.911*Re_D**0.385*Pr**(1/3)
    Nu_D = np.array([Nu_D_1[0], Nu_D_2[1]])
    h_conv = Nu_D * k / D
    h_rad = epsilon*sigma*(T_f + T_inf + 273.15*2)*((T_f+273.15)**2 + (T_inf+273.15)**2)
    h_measured = np.array([18.72, 48.89])
    h_predicted = h_conv + h_rad

    VALUES.clear()
    VALUES['T_f'] = T_f
    VALUES['U_air'] = U_air
    VALUES['Re_D'] = Re_D
    VALUES['Gr_D'] = Gr_D
    VALUES['Pr'] = Pr
    VALUES['Nu_D'] = Nu_D
    VALUES['h_conv'] = h_conv
    VALUES['h_rad'] = h_rad
    VALUES['h_predicted'] = h_predicted
    VALUES['h_measured'] = h_measured

    # print(VALUES)

    print_values('2')
    
def calculations():
    # exercise_a_calculations()
    # figure_1_plot()
    # figure_2_plot()
    figure_3_plot()
    figure_4_plot()
    # exercise_b_calculations()
    # exercise_d_calculations()
    # table_2_calculations()
    return

calculations()

print("Saved all figures.")

