import pandas as pd
import math
import matplotlib.pyplot as plt

###############################################################################
# HELPER FUNCTIONS & CONSTANTS
###############################################################################

# returns the time value from a given sample number
def sample_to_time(n: int) -> int:
    return (n-1) * 10

def get_average(data: list[float], ss_start:int):
    return sum(data[ss_start:]) / len(data[ss_start:])

def temp_err(T : float) -> float:
    return 2.2 if T*0.0075 < 2.2 else T*0.0075

def linear_interpolation(x1 : float, y1 : float, x2 : float, y2 : float, x3 : float) -> float:
    assert(x1 <= x3 <= x2)
    assert(y1 <= y2)
    return y1 + (y2 - y1) * (x3 - x1) / (x2 - x1)

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

###############################################################################
# EXERCISE A : CYLINDER TEMPERATURE VS TIME
###############################################################################

# Data Parsing
xls = pd.ExcelFile('./Data/C5_A.xlsx')
data = []
labels = ["4.5V", "9V", "13.5V", "18V", "Steady State Markers"]
for i in range(4):
    data.append(pd.read_excel(xls, labels[i]))

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("Cylinder Temperature (C) vs Time (s)")
cyl_temps = []
cyl_temps_err = []
times = []
steady_states = [124, 74, 40, 28]
for i in range(len(data)):
    sheet = data[i]
    # Line plots
    times.append(list(map(sample_to_time, sheet["Sample Number "])))
    cyl_temps.append(list(sheet["Heater Temp T10 [°C]"]))
    cyl_temps_err.append(list(map(temp_err,
                         sheet["Heater Temp T10 [°C]"])))
    plt.plot(times[i],cyl_temps[i])

for i in range(len(data)):
    # Steady state plots
    ss_sample = steady_states[i]
    plt.plot(sample_to_time(ss_sample), cyl_temps[i][ss_sample], 
        marker="o", markersize=5, color=COLORS[i], 
        markeredgecolor="black", markerfacecolor="black", )
    plt.errorbar(times[i][::3], cyl_temps[i][::3], yerr=cyl_temps_err[i][::3], ls="None", color=COLORS[i], capsize=5)
    # print(cyl_temps[i][ss_sample])

plt.xlabel('Time (s)')
plt.ylabel('Cylinder Temperature (C)')
plt.legend(labels)
plt.savefig('./Figure 1.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE A : HEAT TRANSFER COEFFICIENT VS HEATER TEMPERATURE
###############################################################################

# Data Parsing
xls = pd.ExcelFile('./Data/C5_A.xlsx')
data = []
labels = ["4.5V", "9V", "13.5V", "18V", "Steady State Markers"]
for i in range(4):
    data.append(pd.read_excel(xls, labels[i]))

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("Free Heat Transfer Coefficient (W/m^2*K) vs Heater Temperature (C)")
heater_temps = [83.0078, 202.148, 316.528, 430.542]
h_free = [21.7597, 26.4906, 29.5052, 31.8077]
h_free_err = [4.35739, 5.30474, 5.90841, 6.36949]
plt.plot(heater_temps, h_free,"-o", color=COLORS[1]) 
for i in range(len(data)):
    sheet = data[i]
    heater_temps.append(list(sheet["Heater Temp T10 [°C]"]))
    plt.errorbar(heater_temps[i], h_free[i], yerr=h_free_err[i], color=COLORS[1], capsize=5)

plt.xlabel('Heater Temperature (C)')
plt.ylabel('Free Heat Transfer Coefficient (W/m^2*K)')
plt.savefig('./Figure 2.png', bbox_inches='tight')
# plt.show() 

###############################################################################
# EXERCISE A : HEAT TRANSFER VS HEATER TEMPERATURE
###############################################################################

# Data Parsing
labels = ["q_elec", "q_free", "q_radiation", "q_total", "Steady State Markers"]

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("q_elec, q_free, q_radiation, q_total (W) vs Heater Temperature (C)")
heater_temps = [83.0078, 202.148, 316.528, 430.542]
q_elec = [3.08, 12.74, 28.14, 50.4]
q_elec_err = [0.445533, 0.920706, 1.35636, 1.82165]
q_free = [2.87202, 10.4106, 18.9937, 28.4238]
q_free_err = [0.594787, 2.09523, 3.81411, 5.70565]
q_rad = [0.00559072, 0.197769, 1.18902, 4.07017]
q_rad_err = [0.000624482, 0.0108233, 0.0531659, 0.181991]
q_tot = [2.87761, 10.6084, 20.1827, 32.494]
q_tot_err = [0.594787, 2.09525, 3.81448, 5.70855]
plt.errorbar(heater_temps, q_elec, yerr=q_elec_err, color=COLORS[0], capsize=5)
plt.errorbar(heater_temps, q_free, yerr=q_free_err, color=COLORS[1], capsize=5)
plt.errorbar(heater_temps, q_rad, yerr=q_rad_err, color=COLORS[2], capsize=5)
plt.errorbar(heater_temps, q_tot, yerr=q_tot_err, color=COLORS[3], capsize=5)
plt.plot(heater_temps, q_elec,"-o", color=COLORS[0])
plt.plot(heater_temps, q_free,"-o", color=COLORS[1])
plt.plot(heater_temps, q_rad,"-o", color=COLORS[2])
plt.plot(heater_temps, q_tot,"-o", color=COLORS[3]) 

plt.legend(labels[:-1])
plt.xlabel("Surface Temperature (C)")
plt.ylabel('Heat Transfer (W)')
plt.savefig('./Figure 3.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : 4.5V CYLINDER TEMPERATURE VS TIME
###############################################################################

# Data Parsing
xls = pd.ExcelFile('./Data/4.5V_C5_B.xlsx')
data = []
labels = ["1.0 ms", "2.5 ms", "4 ms", "5.5 ms", "7 ms", "Steady State Markers"]
for i in range(5):
    data.append(pd.read_excel(xls, labels[i]))

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("4.5V Cylinder Temperature vs. Time")
cyl_temps = []
air_temps = []
cyl_temps_err = []
times = []
steady_states = [50, 27, 25, 20, 16]
for i in range(len(data)):
    sheet = data[i]
    # Line plots
    times.append(list(map(sample_to_time, sheet["Sample Number "])))
    cyl_temps.append(list(sheet["Heater Temp T10 [°C]"]))
    air_temps.append(list(sheet["Duct Temp T9 [°C]"]))
    plt.plot(times[i],cyl_temps[i], linewidth=1)
    cyl_temps_err.append(list(map(temp_err,
                         sheet["Heater Temp T10 [°C]"])))
    plt.errorbar(times[i][::3], cyl_temps[i][::3], yerr=cyl_temps_err[i][::3], ls="None", color=COLORS[i], capsize=5)

labels = ["1.0 m/s", "2.5 m/s", "4 m/s", "5.5 m/s", "7 m/s", "Steady State Markers"]
# print("Steady State 4.5V: ")
# Steady state plots
for i in range(len(data)):
    ss_sample = steady_states[i]
    # print(air_temps[i][ss_sample])
    plt.plot(sample_to_time(ss_sample), cyl_temps[i][ss_sample] + 0.5, 
        marker="s", markersize=3, 
        markeredgecolor="black", markerfacecolor="black")

plt.xlabel('Time (s)')
plt.ylabel('Temperatures (C)')
plt.legend(labels)
plt.savefig('./Figure 4.1.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : 18V CYLINDER TEMPERATURE VS TIME
###############################################################################

# Data Parsing
xls = pd.ExcelFile('./Data/18V_C5_B.xlsx')
data = []
labels = ["1.0 ms", "2.5 ms", "4 ms", "5.5 ms", "7 ms", "Steady State Markers"]
for i in range(5):
    data.append(pd.read_excel(xls, labels[i]))

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("18V Cylinder Temperature vs. Time")
cyl_temps = []
air_temps = []
cyl_temps_err = []
times = []
steady_states = [30, 27, 24, 20, 18]
for i in range(len(data)):
    sheet = data[i]
    # Line plots
    times.append(list(map(sample_to_time, sheet["Sample Number "])))
    cyl_temps.append(list(sheet["Heater Temp T10 [°C]"]))
    air_temps.append(list(sheet["Duct Temp T9 [°C]"]))
    plt.plot(times[i],cyl_temps[i], linewidth=1)
    cyl_temps_err.append(list(map(temp_err,
                         sheet["Heater Temp T10 [°C]"])))
    plt.errorbar(times[i][::3], cyl_temps[i][::3], yerr=cyl_temps_err[i][::3], ls="None", color=COLORS[i], capsize=5)

# print("Steady State 18V: ")
# Steady state plots
for i in range(len(data)):
    ss_sample = steady_states[i]
    # print(air_temps[i][ss_sample])
    plt.plot(sample_to_time(ss_sample), cyl_temps[i][ss_sample], 
        marker="s", markersize=3, 
        markeredgecolor="black", markerfacecolor="black")

labels = ["1.0 m/s", "2.5 m/s", "4 m/s", "5.5 m/s", "7 m/s", "Steady State Markers"]
plt.xlabel('Time (s)')
plt.ylabel('Temperatures (C)')
plt.legend(labels)
plt.savefig('./Figure 4.2.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : 18V CYLINDER TEMPERATURE VS AIR VELOCITY
###############################################################################

# Data Parsing
xls_1 = pd.ExcelFile('./Data/4.5V_C5_B.xlsx')
xls_2 = pd.ExcelFile('./Data/18V_C5_B.xlsx')
data = [[], []] 
labels = ["1.0 ms", "2.5 ms", "4 ms", "5.5 ms", "7 ms", "Steady State Markers"]
for i in range(len(labels) - 1):
    data[0].append(pd.read_excel(xls_1, labels[i]))
    data[1].append(pd.read_excel(xls_2, labels[i]))

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("Cylinder Temperature (C) vs. Air Velocity (m/s)")
steady_states_1 = [50, 27, 25, 20, 16]
cyl_temps_1 = [list(data[0][i]["Heater Temp T10 [°C]"])[steady_states_1[i]] 
                for i in range(len(steady_states_1))]
# print(cyl_temps_1)
cyl_temps_err_1 = list(map(temp_err, cyl_temps_1))
steady_states_2 = [30, 27, 24, 20, 18]
cyl_temps_2 = [list(data[1][i]["Heater Temp T10 [°C]"])[steady_states_2[i]]
                for i in range(len(steady_states_2))]
# print(cyl_temps_2)
cyl_temps_err_2 = list(map(temp_err, cyl_temps_2))
air_vels = [1.0, 2.5, 4.0, 5.5, 7.0]
plt.plot(air_vels, cyl_temps_1, color = COLORS[0])
plt.plot(air_vels, cyl_temps_2, color = COLORS[1])
for i in range(5):
    plt.errorbar(air_vels[i], cyl_temps_1[i], yerr=cyl_temps_err_1[i], ls="None", color=COLORS[0], capsize=5)
    plt.errorbar(air_vels[i], cyl_temps_2[i], yerr=cyl_temps_err_2[i], ls="None", color=COLORS[1], capsize=5)

plt.xlabel('Air Velocity (m/s)')
plt.ylabel('Temperatures (C)')
plt.legend(["4.5V", "18V"])
plt.savefig('./Figure 5.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : FORCED CONVECTION COEFFICIENT VS AIR VELOCITY
###############################################################################

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("Forced Convection Heat Transfer Coefficient (W/m^2*K) vs. Air Velocity (m/s)")
h_conv_1 = [36.6189, 54.4605, 67.7681, 79.0076, 92.0019]
h_conv_err_1 = [7.33292, 10.9057, 13.5706, 15.8213, 18.4234]
h_conv_2 = [36.6121, 54.4603, 67.77, 78.9744, 91.932]
h_conv_err_2 = [7.33156, 10.9057, 13.5709, 15.8146, 18.4094]
air_vels = [1.0, 2.5, 4.0, 5.5, 7.0]

q_elec = [52.2, 52.2, 52.2, 52.2, 52.2]
q_conv = [2.42808, 2.22115, 2.15755, 2.13769, 2.21419]
diff_t = [q_elec[i] - q_conv[i] for i in range(5)]
diff_b = list(map(lambda x: x * 0.01 * 0.07 * math.pi, 
    [286.905, 243.708, 203.589, 178.093, 158.693]))
h_meas = [diff_t[i] / diff_b[i] for i in range(5)]

plt.plot(air_vels, h_conv_1, color = COLORS[0])
plt.plot(air_vels, h_conv_2, color = COLORS[1])
plt.plot(air_vels, h_meas, color = COLORS[2])
for i in range(5):
    plt.errorbar(air_vels[i], h_conv_1[i], yerr=h_conv_err_1[i], ls="None", color=COLORS[0], capsize=5)
    plt.errorbar(air_vels[i], h_conv_2[i], yerr=h_conv_err_2[i], ls="None", color=COLORS[1], capsize=5)

plt.xlabel('Air Velocity (m/s)')
plt.ylabel('Temperatures (C)')
plt.legend(["4.5V", "18V", "18V h_meas"])
plt.savefig('./Figure 6.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : 4.5V HEAT TRANSFER VS HEATER TEMPERATURE
###############################################################################

# Data Parsing
labels = ["q_elec", "q_conv", "q_radiation", "q_total", "Steady State Markers"]

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("4.5V q_elec, q_conv, q_radiation, q_total (W) vs Air Velocity (m/s)")
air_vels = [1.0, 2.5, 4.0, 5.5, 7.0]
q_elec = [3.08, 3.08, 3.08, 3.08, 3.08]
q_elec_err = [0.445533, 0.445533, 0.445533, 0.445533, 0.445533]
q_conv = [2.42808, 2.22115, 2.15755, 2.13769, 2.21419]
q_conv_err = [0.547529, 0.580675, 0.63414, 0.689877, 0.770287]
q_rad = [0.000900993, 0.000305041, 0.000193563, 0.000139718, 0.000111956]
q_rad_err = [0.00015834, 7.35899e-05, 5.49998e-05, 4.47211e-05, 3.91201e-05]
q_tot = [2.42898, 2.22145, 2.15774, 2.13783, 2.2143]
q_tot_err = [0.547529, 0.580675, 0.63414, 0.689877, 0.770287]
plt.errorbar(air_vels, q_elec, yerr=q_elec_err, color=COLORS[0], capsize=5)
plt.errorbar(air_vels, q_conv, yerr=q_conv_err, color=COLORS[1], capsize=5)
plt.errorbar(air_vels, q_rad, yerr=q_rad_err, color=COLORS[2], capsize=5)
plt.errorbar(air_vels, q_tot, yerr=q_tot_err, color=COLORS[3], capsize=5)
plt.plot(air_vels, q_elec,"-o", color=COLORS[0])
plt.plot(air_vels, q_conv,"-o", color=COLORS[1])
plt.plot(air_vels, q_rad,"-o", color=COLORS[2])
plt.plot(air_vels, q_tot,"-o", color=COLORS[3])

plt.legend(labels[:-1])
plt.xlabel("Air Velocity (m/s)")
plt.ylabel('Heat Transfer Coefficient (W/m^2*K)')
plt.savefig('./Figure 7.1.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : 18V HEAT TRANSFER VS HEATER TEMPERATURE
###############################################################################

# Data Parsing
labels = ["q_elec", "q_conv", "q_radiation", "q_total", "Steady State Markers"]

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("18V q_elec, q_conv, q_radiation, q_total (W) vs Air Velocity (m/s)")
air_vels = [1.0, 2.5, 4.0, 5.5, 7.0]
q_elec = [52.2, 52.2, 52.2, 52.2, 52.2]
q_elec_err = [1.82321, 1.82321, 1.82321, 1.82321, 1.82321]
q_conv = [23.0999, 29.1876, 30.3417, 30.9301, 32.0828]
q_conv_err = [4.63877, 5.86409, 6.10128, 6.22512, 6.46342]
q_rad = [1.08103, 0.594017, 0.310735, 0.193506, 0.129954]
q_rad_err = [0.0483369, 0.027817, 0.0158794, 0.0106266, 0.00762003]
q_tot = [24.1809, 29.7816, 30.6524, 31.1236, 32.2127]
q_tot_err = [4.63902, 5.86416, 6.1013, 6.22513, 6.46342]
plt.errorbar(air_vels, q_elec, yerr=q_elec_err, color=COLORS[0], capsize=5)
plt.errorbar(air_vels, q_conv, yerr=q_conv_err, color=COLORS[1], capsize=5)
plt.errorbar(air_vels, q_rad, yerr=q_rad_err, color=COLORS[2], capsize=5)
plt.errorbar(air_vels, q_tot, yerr=q_tot_err, color=COLORS[3], capsize=5)
plt.plot(air_vels, q_elec,"-o", color=COLORS[0])
plt.plot(air_vels, q_conv,"-o", color=COLORS[1])
plt.plot(air_vels, q_rad,"-o", color=COLORS[2])
plt.plot(air_vels, q_tot,"-o", color=COLORS[3])

plt.legend(labels[:-1])
plt.xlabel("Air Velocity (m/s)")
plt.ylabel('Heat Transfer Coefficient (W/m^2*K)')
plt.savefig('./Figure 7.2.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : 18V RELATIVE DIFFERENCE
###############################################################################

# Data Parsing
labels = ["q_elec", "q_conv", "q_radiation", "q_total", "Steady State Markers"]

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("18V Total vs. Electric Heat Transfer Relative Difference (%) vs Air Velocity (m/s)")
air_vels = [0.0, 1.0, 2.5, 4.0, 5.5, 7.0]
q_elec = [50.4, 52.2, 52.2, 52.2, 52.2, 52.2]
q_tot = [31.6159, 24.1809, 29.7816, 30.6524, 31.1236, 32.2127]
rel_diff = [100*abs((q_elec[i]-q_tot[i])/q_elec[i]) for i in range(6)]
plt.plot(air_vels, rel_diff,"-o", color=COLORS[0])

plt.xlabel("Air Velocity (m/s)")
plt.ylabel('Relative Percentage Difference (%)')
plt.savefig('./Figure 8.png', bbox_inches='tight')
# plt.show()

###############################################################################
# EXERCISE B : 18V RELATIVE DIFFERENCE (ALL DATA)
###############################################################################

# Line Plotting environment
plt.figure(figsize=(25,10))
plt.title("18V Total vs. Electric Heat Transfer Relative Difference (%) vs Air Velocity (m/s)")
T_s_a = [83.0078125, 202.1484375, 316.5283203125, 430.5419921875]
T_s_b_1 = [52.978515625, 41.015625, 37.109375, 34.5458984375, 32.958984375]
T_s_b_2 = [309.08203125, 266.11328125, 226.318359375, 201.0498046875, 182.0068359375]
q_elec_a = [3.08, 12.74, 28.14, 50.4]
q_tot_a = [2.47967, 9.97067, 19.4096, 31.6159]
q_elec_b_1 = [3.08, 3.08, 3.08, 3.08, 3.08]
q_tot_b_1 = [2.42898, 2.22145, 2.15774, 2.13783, 2.2143]
q_elec_b_2 = [52.2, 52.2, 52.2, 52.2, 52.2]
q_tot_b_2 = [24.1809, 29.7816, 30.6524, 31.1236, 32.2127]
rel_diff_a = [100*abs((q_elec_a[i]-q_tot_a[i])/q_elec_a[i]) for i in range(4)]
rel_diff_b_1 = [100*abs((q_elec_b_1[i]-q_tot_b_1[i])/q_elec_b_1[i]) for i in range(5)]
rel_diff_b_2 = [100*abs((q_elec_b_2[i]-q_tot_b_2[i])/q_elec_b_2[i]) for i in range(5)]
plt.scatter(T_s_a, rel_diff_a, color=COLORS[0])
plt.scatter(T_s_b_1, rel_diff_b_1, color=COLORS[1])
plt.scatter(T_s_b_2, rel_diff_b_2, color=COLORS[2])

plt.legend(["Free Flow 4.5V", "Forced Flow 4.5V", "Forced Flow 18V"])
plt.xlabel("Cylinder Temperature (C)")
plt.ylabel('Relative Percentage Difference (%)')
plt.savefig('./Figure 9.png', bbox_inches='tight')
# plt.show()

print("Saved all figures.")

