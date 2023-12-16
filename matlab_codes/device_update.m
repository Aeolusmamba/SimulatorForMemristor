G_min = 0;
G_max = 40e-9;
P_max = 50;

% linear: P 和G的函数
P = 0: 2 * P_max;
G = zeros(size(P));
for i = 1: length(P)
    if P(i) <= P_max
        G(i) = (G_max - G_min) / P_max * P(i) + G_min;
    else
        G(i) = (G_min - G_max) / P_max * P(i) - 2 * (G_min - G_max);
    end
end

figure(1);
plot(P, G);
title('linear programming')
xlabel('puluse #') 
ylabel('Conductance (S)') 

axis([-5 120 -5e-9 50e-9]);
grid on;

% nonlinear: P 和G的函数

% nonlinearity
nonlinearityLTP = 5;
nonlinearityLTD = -5;
B = (G_max - G_min) / (1 - exp(-P_max / nonlinearityLTP));
% B_LTD = (G_max - G_min) / (1 - exp(-P_max / nonlinearityLTD));
figure(2);
fplot(@(P) B * (1 - exp(-P / nonlinearityLTP)) + G_min, [0 P_max]);
hold on;
fplot(@(P) -B * (1 - exp((P - P_max) / nonlinearityLTD)) + G_max, [P_max 2*P_max])
hold off;
title('nonlinear programming')
xlabel('puluse #') 
ylabel('Conductance (S)') 
axis([-5 120 -5e-9 50e-9]);
grid on;
