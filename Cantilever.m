% Physics-Informed Neural Network (PINN) for Cantilever Beam Deflection
% Governing equation: EI * y''(x) = P(L - x)
% Analytical solution: y(x) = (P/(6EI))(3Lx^2 - x^3)

% Clear workspace
clear; clc;

%% Beam parameters
L = 1;          % Length (m)
E = 200e9;      % Young's modulus (Pa)
I = 1e-6;       % Moment of inertia (m^4)
P = 1000;       % Tip load (N)

% Analytical solution
analytical_y = @(x) (P/(6*E*I)) * (3*L*x.^2 - x.^3);

%% Generate synthetic training data (with noise)
num_data_points = 10;                % Sparse measurements
x_data = linspace(0, L, num_data_points)';  
y_data = analytical_y(x_data) + 0.1*randn(size(x_data)); % Add 10% noise

%% Collocation points for physics loss (no data required)
num_colloc_points = 100;
x_colloc = linspace(0, L, num_colloc_points)';

%% Define the neural network
layers = [
    featureInputLayer(1, 'Name', 'input')    % Input: x-coordinate
    fullyConnectedLayer(20, 'Name', 'fc1')    % Hidden layer 1
    tanhLayer('Name', 'tanh1')                % Activation
    fullyConnectedLayer(20, 'Name', 'fc2')    % Hidden layer 2
    tanhLayer('Name', 'tanh2')                % Activation
    fullyConnectedLayer(1, 'Name', 'output')  % Output: deflection y(x)
];

net = dlnetwork(layers);  % Create network

%% Training parameters
numEpochs = 3000;         % Training iterations
initialLearnRate = 0.001; % Learning rate
lambda = 0.5;             % Weight for physics loss

% Convert data to dlarray (for automatic differentiation)
x_train_dl = dlarray(x_data', 'CB');  % Channel x Batch format
y_train_dl = dlarray(y_data', 'CB');

% Adam optimizer
averageGrad = [];
averageSqGrad = [];

%% Training loop
for epoch = 1:numEpochs
    % Evaluate data loss
    y_pred = forward(net, x_train_dl);
    loss_data = mse(y_pred, y_train_dl);
   
    % Evaluate physics loss at collocation points
    x_phys_dl = dlarray(x_colloc', 'CB');
    y_phys = forward(net, x_phys_dl);
   
    % Compute second derivative d2y/dx2 using automatic differentiation
    dy_dx = dlgradient(sum(y_phys), x_phys_dl);      % First derivative
    d2y_dx2 = dlgradient(sum(dy_dx), x_phys_dl);     % Second derivative
   
    % Residual of the beam equation: EI*d2y/dx2 - P(L - x)
    residual = E*I * d2y_dx2 - P*(L - x_phys_dl);
    loss_physics = mean(residual.^2);
   
    % Total loss (data + physics)
    total_loss = loss_data + lambda * loss_physics;
   
    % Compute gradients
    gradients = dlgradient(total_loss, net.Learnables);
   
    % Update network using Adam optimizer
    [net, averageGrad, averageSqGrad] = adamupdate(...
        net, gradients, averageGrad, averageSqGrad, epoch, initialLearnRate);
   
    % Display progress
    if mod(epoch, 500) == 0
        fprintf('Epoch %d, Total Loss: %.4e\n', epoch, extractdata(total_loss));
    end
end

%% Predict and plot results
x_test = linspace(0, L, 100)';
x_test_dl = dlarray(x_test', 'CB');
y_pred_dl = forward(net, x_test_dl);
y_pred = extractdata(y_pred_dl)';

% Analytical solution for comparison
y_analytical = analytical_y(x_test);

figure;
plot(x_test, y_analytical, 'b-', 'LineWidth', 2); hold on;
plot(x_test, y_pred, 'r--', 'LineWidth', 2);
scatter(x_data, y_data, 50, 'k', 'filled');
xlabel('Position (x)');
ylabel('Deflection (y)');
legend('Analytical Solution', 'PINN Prediction', 'Noisy Training Data');
title('Cantilever Beam Deflection Prediction');
grid on;