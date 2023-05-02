% Code for Home Lab
% 24321 A
% by Group A1

%% initialize data
T_inf = (72-32)*(5/9); % room temperature, Celsius 
rho = 1000; % density, kg/m^3
c = 4184; % specific heat, J/kg*K
V = 0.00118294; %  volume, m^3
h = 0.085725; % height of pot, m
d_p = 0.2413; % diameter of pot, m
A_p = pi()*d_p*h; % surface area of pot wall, m^2
A_w = (pi()*d_p.^2)/4; % water surface area, m^2
% d_b = ; % diameter of burner, m
q_b = 3000; % burner rating, W (13000 Btu/hr)

time = [0, 30, 60, 90, 120, 150, 180, 200]; % s
T = [23.5, 27.3, 32.0, 39.8, 48.9, 56.5, 64.8, 70.8]; % converted to Celsius on Excel
err_T = 1.25.*ones(1,length(T)); %error in temperature measurements

%% Question 4
% 4a
h_L = [34646.62109 27499.06704 31672.90369 36081.19362 39620.63755 41915.83591 44006.02384 45324.25268]; % natural convection coefficients, pot wall
avg_h_L = mean(h_L);
h_D = [10821.64335 14490.2705  16876.49147 19388.67444 21397.30413 22691.84987 23862.79816 24596.81223]; % natural convection coefficients, water surface
avg_h_D = mean(h_D);

a = (1./(rho*V*c))*((avg_h_L*A_p) + (avg_h_D*A_w));
b = @(q) q./(rho*V*c);
Tw = @(q) T_inf + (T(1)-T_inf).*(exp(-a.*time) + (((q./(rho*V*c))./a)./(T(1)-T_inf)).*(1 - exp(-a.*time)));
err_Tw = [0.5 0.6 0.3 0.7 0.4 0.3 0.4 0.6];
% err = 0.60; % error bound
% q_in = 0; % initialize q_in that minimizes error
% mmse = 0; % initialize minimum MSE
% for i = 1:10000 % iterates through q = 1:10000
%     mse = immse(Tw(i),T); % calculates mean squared error between two vectors
%     if mse < err % checks if error is within our bounds
%         mmse = mse;
%         q_in = i
%     end
% end
[MSE, q_in] = minimize(Tw, 1, 10000, T, 0.1);
disp(q_in);
T_calc = Tw(q_in);
disp(T_calc);
% 4b
figure('Name','Question 4')
errorbar(time,T,err_T,'b-o','linewidth',2)
hold on
errorbar(time,T_calc,err_Tw,'r-','linewidth',2)
title('Fitted T_{water} vs Experimental T_{water}','fontsize',16)
xlabel('Time (s)','fontsize',14)
ylabel('Temperature (Celsius)','fontsize',14)
legend('Experimental Temperatures','Fitted T_{water}(t)','location','southeast','fontsize',12)
hold off

%% 6
q_pot = [0.2361204892 0.60632366 1.295125691 2.603618015 4.304419765 5.830844677 7.585853508 8.90329162];
err_q_pot = [6.075228836	12.94552531	19.04217495	26.06098468	33.97184359 40.1325843 47.891023 55.918234];
q_wat = [0.04088352685 0.1771109438 0.3825503037 0.7755804237 1.288648776 1.749869218 2.280321272 2.67843506];
err_q_wat = [0.09202101499	0.5035792617	0.9548329459	1.461212602	2.008403874 2.5128944 3.0129375 3.553123];
q_watabs = [0	20.42876654	72.92761034	185.8908971	359.5141984	554.6615119	795.4425769	983.3818615];
err_q_watabs = [5 5 5 5 5 5 5 5];
figure('Name','Question 6')
errorbar(time,q_b.*ones(1,length(time)),0.*ones(1,length(time)),'b','linewidth',2)
hold on
errorbar(time,q_in.*ones(1,length(time)),10.*ones(1,length(time)),'r-','linewidth',2)
errorbar(time,q_pot,err_q_pot,'linewidth',2)
errorbar(time,q_wat,err_q_wat,'linewidth',2)
errorbar(time,q_watabs,err_q_watabs,'linewidth',2)
title('Heat Flows vs Time','fontsize',16)
xlabel('Time (s)','fontsize',14)
ylabel('Watts (W)','fontsize',14)
legend('q_{burner}','q_{in}','q_{pot,surr}','q_{wat,surr}','q_{wat,abs}','location','best','fontsize',12)
hold off

%% Functions
function [MSE, xOpt] = minimize(f, x_l, x_u, y_exp, e_max) 

% Takes a function f and minimizes its MSE with respect to experimental
% data (y_exp) between lower and upper bounds(x_l and x_u) with an error 
% less than e_max
% Syntax- mseGoldenOpt(f, x_l, x_u, y_exp, e_max)

i_max = 1000;
i = 0;

phi = (1 + sqrt(5))/2;
d = (phi - 1)*(x_u - x_l);

x1 = x_l + d;
x2 = x_u - d;

MSE = 0;
xOpt = 0;

len = numel(y_exp);

while d > e_max && i < i_max
    
    f1 = zeros(1, len);
    f2 = zeros(1, len);
    
    f1 = f(x1);
    f2 = f(x2);
    
    MSE1 = immse(f1, y_exp);
    MSE2 = immse(f2, y_exp);
    
    if MSE1 < MSE2
        xOpt = x1;
        x_l = x2;
        x2 = x1;
        d = (phi - 1)*(x_u - x_l);
        x1 = x_l + d;
        MSE = MSE1;
    else
        xOpt = x2;
        x_u = x1;
        x1 = x2;
        d = (phi - 1)*(x_u - x_l);
        x2 = x_u - d;
        MSE = MSE2;
    end
    
    i = i + 1;
    
end

end