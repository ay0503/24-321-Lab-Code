import math

def getH(T_exp, k):
    # Constants
    d = 0.0127 # meters
    P = math.pi * d
    A = math.pi/4*d**2
    T_inf = 21.7 # Celcius
    
    T_b = T_exp[4] # heated base temperature 
    x = [0, 3, 6, 9, 12] #distance from base in inches
    x = [round(i*0.0254, 4) for i in x] #convert to inches
    L = x[4] # length of entire fin
    
    FMin=10000
    hMin=None
    # h values range from 0 to 500 with increment of 0.1
    for i in range(1, 10000):
        h = i/10 # w/mk
        m = math.sqrt((h*P)/(k*A))
        F=0
        for j in range(5):
            T = T_inf + (T_b-T_inf)*((math.cosh(m*(L-x[j]))+(h/(m*k))\
                *math.sinh(m*(L-x[j])))\
                /(math.cosh(m*L)+(h/(m*k))*math.sinh(m*L)))
            F+=(T_exp[j]-T)**2
        if F < FMin:
            FMin=F
            hMin=h
    if FMin == 10000: print('Invalid Result: Max Iteration Reached')    
    return hMin

#Brass, forced
T_exp_bfo = [23.5698, 24.2858, 26.1338, 32.4907,57.869]
kBrass = 116 #W/mk
print("Brass, forced: "+str(getH(T_exp_bfo, kBrass)))

#Copper, forced
T_exp_cfo = [32.4202, 34.7091, 36.3617, 40.8295, 48.889]
kCopper = 401 #W/mk
print("Copper, forced: "+str(getH(T_exp_cfo, kCopper)))

#Steel, forced
T_exp_sfo = [23.0789, 23.0422, 23.4398, 24.6957, 56.0011]
kSteel = 14.9 #W/mk
print("Steel, forced: "+str(getH(T_exp_sfo, kSteel)))

#Aluminum, forced
T_exp_afo = [25.0694, 25.3451, 26.7059, 29.9053, 40.859]
kAlu = 167 #W/mk
print("Aluminum, forced: "+str(getH(T_exp_afo, kAlu)))

#Brass, free
T_exp_bfr = [30.4336, 32.6462, 38.6185, 51.4694, 76.5699]
print("Brass, free: "+str(getH(T_exp_bfr, kBrass)))

#Copper, free
T_exp_cfr = [46.2017, 50.5833, 53.6397, 60.3475, 68.9532]
print("Copper, free: "+str(getH(T_exp_cfr, kCopper)))

#Steel, free
T_exp_sfr = [22.2945, 22.5085, 24.3151, 34.2729, 82.7978]
print("Steel, free: "+str(getH(T_exp_sfr, kSteel)))

#Aluminum, free
T_exp_afr = [28.5176, 29.329, 32.6868, 39.2744, 51.1504]
print("Aluminum, free: "+str(getH(T_exp_afr, kAlu)))