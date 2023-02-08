%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXERCISE A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp("EXERCISE A DATA: ")

% Constants & Properties
g = 9.81; % (m/s^2)
D = 0.01; % (m)
L = 0.07; % (m)
A_s = pi * D * L; % (m^2)
D_err = 0.0001; % (m)
L_err = 0.0001; % (m)
sigma = 5.67e-8; % (W/m*K^4)
epsilon = 0.95;
epsilon_err = 0.03;
nu = [1.554639e-5; 1.5586852e-5; 1.5618641e-5; 1.5653319e-5]; % (m^2/s)
k = [25.99112e-3; 26.02752e-3; 26.05608e-3; 26.08728e-3]; % (W/m*K)
Pr = [0.7080037929687499; 0.707885599609388; 0.707792733398444; 0.7076914248046939];
alpha = [21.9903876; 22.050381599999998; 22.097531999999994; 22.148972399999998].*1e-6;

% Calculations : Data Parsed and Manipulated Using Python
V = [4.4; 9.1; 13.4; 18.0]; % (V)
V_err = [0.1; 0.1; 0.1; 0.1]; % (V)
I = [0.7; 1.4; 2.1; 2.8]; % (A)
I_err = [0.1; 0.1; 0.1; 0.1]; % (A)
T_air = [22.9892578125; 23.44384765625; 23.801025390625; 24.190673828125]; % (C)
T_air_err = [2.2; 2.2; 2.2; 2.2];
T_s = [83.0078125; 202.1484375; 316.5283203125; 430.5419921875]; % (C)
T_s_err = T_s .* 0.0075;
T_s_err(1) = 2.2;
T_s_err(2) = 2.2;
dT = T_s - T_air;
dT_4 = T_s.^4 - T_air.^4;
u_air = [0.008433948863636364; 0.010141225961538462; 0.006615423387096774; 0.011564555921052632]; % (m/s)
q_elec = V .* I; % (W)
q_elec_err = sqrt((V .* I_err).^2 + (V_err .* I).^2);
Re_D = u_air .* D ./ nu;
Ra_D = g .* dT .* D^3 ./ (nu .* alpha .* T_air);
Nu_D = (0.6 + (0.387.*Ra_D.^(1/6))./(1 + (0.559./Pr).^(9/16)).^(8/27)).^2;
Nu_D_err = Nu_D * 0.2;
h_free = Nu_D.*k./D; % (W/m^2*K)
h_free_err = sqrt((Nu_D .* k .* D_err / D.^2).^2 + (Nu_D_err .* k ./ D).^2);
q_free = h_free .* A_s .* (T_s - T_air); % (W)
q_free_err = sqrt((pi .* D .* L .* dT .* h_free_err).^2 ...
                + (h_free .* pi .* L .* dT .* D_err).^2 ...
                + (h_free .* pi .* D .* dT .* L_err).^2 ...
                + (h_free .* pi .* D .* L .* T_s_err).^2 ...
                + (-h_free .* pi .* D .* L .* T_air_err).^2); % (W)
q_rad = epsilon .* sigma .* A_s .* (T_s.^4 - T_air.^4); % (W)
q_rad_err = sqrt((pi .* L .* epsilon .* sigma .* dT_4 .* D_err).^2 ...
                + (pi .* D .* epsilon .* sigma .* dT_4 .* L_err).^2 ...
                + (pi .* D .* L .* sigma .* dT_4 .* epsilon_err).^2 ...
                + (4 .* pi .* D .* L .* epsilon .* sigma .* T_s.^3 .* T_s_err).^2 ...
                + (4 .* pi .* D .* L .* epsilon .* sigma .* T_air.^3 .* T_air_err).^2); % (W)
q_tot = q_rad + q_free; % (W)
q_tot_err = sqrt(q_rad_err.^2 + q_free_err.^2);

% Data Printing
printVector(T_air, "T_air");
printVector(T_air_err, "T_air_err");
printVector(T_s, "T_s");
printVector(T_s_err, "T_s_err");
printVector(u_air, "u_air");
printVector(Re_D, "Re_D");
printVector(Ra_D, "Ra_D");
printVector(Nu_D, "Nu_D");
printVector(Nu_D_err, "Nu_D_err");
printVector(h_free, "h_free");
printVector(h_free_err, "h_free_err");
printVector(q_elec, "q_elec");
printVector(q_elec_err, "q_elec_err");
printVector(q_free, "q_free");
printVector(q_free_err, "q_free_err");
printVector(q_rad, "q_rad");
printVector(q_rad_err, "q_rad_err");
printVector(q_tot, "q_tot");
printVector(q_tot_err, "q_tot_err");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXERCISE B 4.5 V
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp("EXERCISE B 4.5V DATA: ");

% Constants & Properties
g = 9.81; % (m/s^2)
D = 0.01; % (m)
L = 0.07; % (m)
A_s = pi * D * L; % (m^2)
D_err = 0.0001; % (m)
L_err = 0.0001; % (m)
sigma = 5.67e-8; % (W/m*K^4)
epsilon = 0.95;
epsilon_err = 0.03;
nu = [1.5531953e-5; 1.550018e-5; 1.5514598e-5; 1.5479888e-5; 1.5459685e-5]; % (m^2/s)
k = [25.97816e-3; 25.9496e-3; 25.96256e-3; 25.93136e-3; 25.9132e-3]; % (W/m*K)
Pr = [0.70804598; 0.7081388; 0.70809668; 0.70819808; 0.7082571];
alpha = [21.968951367187497; 21.92180390625; 21.943234570312498; 21.891800976562497; 21.861798046874995].*1e-6;
V = repelem(4.4, 5);
V_err = 0.1;
I = repelem(0.7, 5);
I_err = 0.1;

% Calculations : Data Parsed and Manipulated Using Python
T_s = [52.978515625; 41.015625; 37.109375; 34.5458984375; 32.958984375];
T_s_err = [2.2; 2.2; 2.2; 2.2; 2.2];
T_air = [22.826904296875; 22.4697265625; 22.632080078125; 22.242431640625; 22.01513671875];
T_air_err = [2.2; 2.2; 2.2; 2.2; 2.2];
U_a = [1.07971964; 2.53112793; 4.045872044; 5.512237549; 7.05078125]; % (m/s)
U_c = 1.22 .* U_a;
dT = T_s - T_air;
dT_4 = T_s.^4 - T_air.^4;
q_elec = V .* I; % (W)
q_elec_err = sqrt((V .* I_err).^2 + (V_err .* I).^2);
Re_D = U_c .* D ./ nu;
Ra_D = g .* dT .* D^3 ./ (nu .* alpha .* T_air);
Gr_D = g .* dT .* D^3 ./ (nu.^2 .* T_air);
ratio = Gr_D ./ Re_D.^2;
Nu_D = 0.683.*Re_D.^0.466.*Pr.^(1/3);
Nu_D(4) = 0.193*Re_D(4)^0.618*Pr(4)^(1/3);
Nu_D(5) = 0.193*Re_D(5)^0.618*Pr(5)^(1/3);
Nu_D_err = Nu_D * 0.2;
h_conv = Nu_D .* k ./ D; % (W/m^2*K)
h_conv_err = sqrt((Nu_D .* k .* D_err / D.^2).^2 + (Nu_D_err .* k ./ D).^2);
q_conv = h_conv .* A_s .* (T_s - T_air); % (W)
q_conv_err = sqrt((pi .* D .* L .* dT .* h_conv_err).^2 ...
                + (h_conv .* pi .* L .* dT .* D_err).^2 ...
                + (h_conv .* pi .* D .* dT .* L_err).^2 ...
                + (h_conv .* pi .* D .* L .* T_s_err).^2 ...
                + (-h_conv .* pi .* D .* L .* T_air_err).^2); % (W)
q_rad = epsilon .* sigma .* A_s .* (T_s.^4 - T_air.^4); % (W)
q_rad_err = sqrt((pi .* L .* epsilon .* sigma .* dT_4 .* D_err).^2 ...
                + (pi .* D .* epsilon .* sigma .* dT_4 .* L_err).^2 ...
                + (pi .* D .* L .* sigma .* dT_4 .* epsilon_err).^2 ...
                + (4 .* pi .* D .* L .* epsilon .* sigma .* T_s.^3 .* T_s_err).^2 ...
                + (4 .* pi .* D .* L .* epsilon .* sigma .* T_air.^3 .* T_air_err).^2); % (W)
q_tot = q_rad + q_conv; % (W)
q_tot_err = sqrt(q_rad_err.^2 + q_conv_err.^2);

% Data Printing
printVector(T_air, "T_air");
printVector(T_air_err, "T_air_err");
printVector(T_s, "T_s");
printVector(T_s_err, "T_s_err");
printVector(u_air, "u_air");
printVector(Gr_D, "Gr_D");
printVector(Re_D, "Re_D");
printVector(Ra_D, "Ra_D");
printVector(Nu_D, "Nu_D");
printVector(ratio, "ratio")
printVector(Nu_D_err, "Nu_D_err");
printVector(h_conv, "h_conv");
printVector(h_conv_err, "h_conv_err");
printVector(q_elec, "q_elec");
printVector(q_elec_err, "q_elec_err");
printVector(q_conv, "q_conv");
printVector(q_conv_err, "q_conv_err");
printVector(q_rad, "q_rad");
printVector(q_rad_err, "q_rad_err");
printVector(q_tot, "q_tot");
printVector(q_tot_err, "q_tot_err");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXERCISE B 18 V
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp("EXERCISE B 18V DATA: ");

% Constants & Properties
g = 9.81; % (m/s^2)
D = 0.01; % (m)
L = 0.07; % (m)
A_s = pi * D * L; % (m^2)
D_err = 0.0001; % (m)
L_err = 0.0001; % (m)
sigma = 5.67e-8; % (W/m*K^4)
epsilon = 0.95;
epsilon_err = 0.03;
nu = [15.474103e-6; 15.49395e-6; 15.523231e-6; 15.543523e-6; 15.575296e-6]; % (m^2/s)
k = [25.92616e-3; 25.9444e-3; 25.97032e-3; 25.98856e-3; 26.01712e-3]; % (W/m*K)
Pr = [0.70821498; 0.7081557; 0.70807146; 0.70801218; 0.70791936];
alpha = [21.883228710937498; 21.913231640624996; 21.95609296875; 21.986095898437497; 22.033243359374996].*1e-6;
V = repelem(18.0, 5);
V_err = 0.1;
I = repelem(2.9, 5);
I_err = 0.1;

% Calculations : Data Parsed and Manipulated Using Python
T_s = [309.08203125; 266.11328125; 226.318359375; 201.0498046875; 182.0068359375];
T_s_err = [2.2; 2.2; 2.2; 2.2; 2.2];
T_s_err(1) = T_s(1)*0.0075;
T_air = [22.177490234375; 22.40478515625; 22.7294921875; 22.956787109375; 23.31396484375];
T_air_err = [2.2; 2.2; 2.2; 2.2; 2.2];
U_a = [1.07971964; 2.53112793; 4.045872044; 5.512237549; 7.05078125]; % (m/s)
U_c = 1.22 .* U_a;
dT = T_s - T_air;
dT_4 = T_s.^4 - T_air.^4;
q_elec = V .* I; % (W)
q_elec_err = sqrt((V .* I_err).^2 + (V_err .* I).^2);
Re_D = U_c .* D ./ nu;
Ra_D = g .* dT .* D^3 ./ (nu .* alpha .* T_air); 
Gr_D = g .* dT .* D^3 ./ (nu.^2 .* T_air);
ratio = Gr_D ./ Re_D.^2;
Nu_D = 0.683.*Re_D.^0.466.*Pr.^(1/3);
Nu_D(4) = 0.193*Re_D(4)^0.618*Pr(4)^(1/3);
Nu_D(5) = 0.193*Re_D(5)^0.618*Pr(5)^(1/3);
Nu_D_err = Nu_D * 0.2;
h_conv = Nu_D .* k ./ D; % (W/m^2*K)
h_conv_err = sqrt((Nu_D .* k .* D_err / D.^2).^2 + (Nu_D_err .* k ./ D).^2);
q_conv = h_conv .* A_s .* (T_s - T_air); % (W)
q_conv_err = sqrt((pi .* D .* L .* dT .* h_conv_err).^2 ...
                + (h_conv .* pi .* L .* dT .* D_err).^2 ...
                + (h_conv .* pi .* D .* dT .* L_err).^2 ...
                + (h_conv .* pi .* D .* L .* T_s_err).^2 ...
                + (-h_conv .* pi .* D .* L .* T_air_err).^2); % (W)
q_rad = epsilon .* sigma .* A_s .* (T_s.^4 - T_air.^4); % (W)
q_rad_err = sqrt((pi .* L .* epsilon .* sigma .* dT_4 .* D_err).^2 ...
                + (pi .* D .* epsilon .* sigma .* dT_4 .* L_err).^2 ...
                + (pi .* D .* L .* sigma .* dT_4 .* epsilon_err).^2 ...
                + (4 .* pi .* D .* L .* epsilon .* sigma .* T_s.^3 .* T_s_err).^2 ...
                + (4 .* pi .* D .* L .* epsilon .* sigma .* T_air.^3 .* T_air_err).^2); % (W)
q_tot = q_rad + q_conv; % (W)
q_tot_err = sqrt(q_rad_err.^2 + q_conv_err.^2);
h_meas = (q_elec - q_rad) ./ (A_s .* (T_s - T_air));

% Data Printing
printVector(T_air, "T_air");
printVector(T_air_err, "T_air_err");
printVector(T_s, "T_s");
printVector(T_s - T_air, "diff");
printVector(T_s_err, "T_s_err");
printVector(u_air, "u_air");
printVector(Re_D, "Re_D");
printVector(Gr_D, "Gr_D");
printVector(Ra_D, "Ra_D");
printVector(Nu_D, "Nu_D");
printVector(Nu_D_err, "Nu_D_err");
printVector(h_conv, "h_conv");
printVector(ratio, "ratio");
printVector(h_conv_err, "h_conv_err");
fprintf("%f", h_meas);
printVector(q_elec, "q_elec");
printVector(q_elec_err, "q_elec_err");
printVector(q_conv, "q_conv");
printVector(q_conv_err, "q_conv_err");
printVector(q_rad, "q_rad");
printVector(q_rad_err, "q_rad_err");
printVector(q_tot, "q_tot");
printVector(q_tot_err, "q_tot_err");