clc
%Constants
d = 0.0127; % meters
L = 0.3048; % meters
P = pi * d;
k = [116 401 14.9 167 116 401 14.9 167]; % W/mk
A = pi/4*d^2;
T_inf = 21.7; % Celcius

Tb=[76.57 68.95 82.8 51.15 57.87 48.89 56 40.86];
h=[23.3 18.72 20.52 37.36 74.87 31.46 45.06 61.81];
m = sqrt(h.*P./(k.*A));

q = sqrt(h.*P.*k.*A).*(Tb - T_inf).*(sinh(m.*L)+(h./m./k).*cosh(m.*L))./...
    (cosh(m.*L)+(h./m./k).*sinh(m.*L));
disp(q)

