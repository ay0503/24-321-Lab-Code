function [MSE, xOpt] = mseGoldenOpt(f, x_l, x_u, y_exp, e_max) 

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