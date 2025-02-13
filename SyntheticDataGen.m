clear all; clc; close all; 

% Define the center (mean) of two different data clusters
xC1 = [20; 9]; % Center of class 1
xC2 = [6; 8];  % Center of class 2

% Define standard deviations (spread) of data along principal axes
sig1_X = 1.6; % Standard deviation along X-axis for class 1
sig2_X = 1.7; % Standard deviation along X-axis for class 2
sig1_Y = 3.5; % Standard deviation along Y-axis for class 1
sig2_Y = 1.4; % Standard deviation along Y-axis for class 2

% Define rotation angles for each dataset (in degrees)
Angle_Set1 = 0; % Rotation for class 1
Angle_Set2 = 0; % Rotation for class 2

% Convert standard deviations into column vectors
sig1 = [sig1_X; sig1_Y]; % Principal axes for class 1
sig2 = [sig2_X; sig2_Y]; % Principal axes for class 2

% Convert angles from degrees to radians
theta1 = (Angle_Set1/180) * pi; % Convert rotation angle for class 1
theta2 = (Angle_Set2/180) * pi; % Convert rotation angle for class 2

% Define rotation matrices for each dataset
R1 = [cos(theta1) -sin(theta1); sin(theta1) cos(theta1)]; % Rotation matrix for class 1
R2 = [cos(theta2) -sin(theta2); sin(theta2) cos(theta2)]; % Rotation matrix for class 2

nPoints = 1000; % Number of data points per class

% Generate random normally distributed data points for each class
X1 = R1 * diag(sig1) * randn(2, nPoints) + diag(xC1) * ones(2, nPoints); % Class 1 data
X2 = R2 * diag(sig2) * randn(2, nPoints) + diag(xC2) * ones(2, nPoints); % Class 2 data

% Transpose data for easier handling in later operations
X_class1 = X1';
X_class2 = X2';

% Scatter plot of generated data points
scatter(X1(1,:), X1(2,:), 'k.', 'LineWidth', 2) % Class 1 in black
hold on
scatter(X2(1,:), X2(2,:), 'b.', 'LineWidth', 2) % Class 2 in blue
hold off
grid on
%%

% Compute the mean of each dataset (class)
Xavg1 = mean(X1,2); % Mean of class 1 (column-wise mean)
Xavg2 = mean(X2,2); % Mean of class 2 (column-wise mean)

% Subtract the mean from the data to center it (mean normalization)
B1 = X1 - Xavg1 * ones(1, nPoints); % Class 1: Zero-mean data
B2 = X2 - Xavg2 * ones(1, nPoints); % Class 2: Zero-mean data

% Perform Principal Component Analysis (PCA) using Singular Value Decomposition (SVD)
[U1, S1, V1] = svd(B1 / sqrt(nPoints), 'econ'); % PCA for class 1
[U2, S2, V2] = svd(B2 / sqrt(nPoints), 'econ'); % PCA for class 2

% Compute principal component angle for class 1
if U1(1,1) > 0 && U1(2,1) > 0 || U1(1,1) < 0 && U1(2,1) < 0  
    angle1 = acosd(U1(1,1)); % Compute angle using cosine inverse
    if angle1 >= 90 
        angle1 = 180 - angle1; % Adjust angle for correct quadrant
    end
end

if U1(1,1) > 0 && U1(2,1) < 0 || U1(1,1) < 0 && U1(2,1) > 0  
    angle1 = asind(U1(1,1)) - 90; % Compute angle using sine inverse
    if angle1 <= -90 
        angle1 = -180 - angle1; % Adjust angle for correct quadrant
    end
end

% Compute principal component angle for class 2
if U2(1,1) > 0 && U2(2,1) > 0 || U2(1,1) < 0 && U2(2,1) < 0  
    angle2 = acosd(U2(1,1)); % Compute angle using cosine inverse
    if angle2 >= 90 
        angle2 = 180 - angle2; % Adjust angle for correct quadrant
    end
end

if U2(1,1) > 0 && U2(2,1) < 0 || U2(1,1) < 0 && U2(2,1) > 0  
    angle2 = asind(U2(1,1)) - 90; % Compute angle using sine inverse
    if angle2 <= -90 
        angle2 = -180 - angle2; % Adjust angle for correct quadrant
    end
end

% Scatter plot of the original data for class 1
figure()
scatter(X1(1,:), X1(2,:), 'k.', 'LineWidth', 2) % Black points for class 1
hold on;

% Generate angles for drawing confidence ellipses
theta = (0:.01:1) * 2 * pi;

% Compute confidence ellipses for class 1
Xstd1 = U1 * S1 * [cos(theta); sin(theta)]; % 1-standard deviation ellipse
plot(Xavg1(1) + Xstd1(1,:), Xavg1(2) + Xstd1(2,:), 'r-') % Red ellipse (1 std)
plot(Xavg1(1) + 2*Xstd1(1,:), Xavg1(2) + 2*Xstd1(2,:), 'r-') % 2 std ellipse
plot(Xavg1(1) + 3*Xstd1(1,:), Xavg1(2) + 3*Xstd1(2,:), 'r-') % 3 std ellipse

% Scatter plot of the original data for class 2
scatter(X2(1,:), X2(2,:), 'k.', 'LineWidth', 2) % Black points for class 2
hold on;

% Compute confidence ellipses for class 2
Xstd2 = U2 * S2 * [cos(theta); sin(theta)]; % 1-standard deviation ellipse
plot(Xavg2(1) + Xstd2(1,:), Xavg2(2) + Xstd2(2,:), 'b-') % Blue ellipse (1 std)
plot(Xavg2(1) + 2*Xstd2(1,:), Xavg2(2) + 2*Xstd2(2,:), 'b-') % 2 std ellipse
plot(Xavg2(1) + 3*Xstd2(1,:), Xavg2(2) + 3*Xstd2(2,:), 'b-') % 3 std ellipse

hold off
grid on % Enable grid for better visualization

%%
% Define the Rosenbrock function
% This function is commonly used as a performance test problem for optimization algorithms.
% It has a global minimum at (1,1) and a characteristic banana-shaped valley.

% Rosenbrock function formula:
% f1(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
f1 = @(x, y) (1 - x).^2 + 100 * (y - x.^2).^2;

% Extract X and Y values from the first two columns of X_class1
x = X_class1(:,1); % X-coordinates
y = X_class1(:,2); % Y-coordinates

% Compute function values and store them in the third column of X_class1
X_class1(:,3) = f1(x, y); % Evaluate Rosenbrock function
% Extract X and Y values from the first two columns of X_class2
x = X_class2(:,1); % X-coordinates
y = X_class2(:,2); % Y-coordinates

% Compute function values and store them in the third column of X_class2
X_class2(:,3) = f1(x, y); % Evaluate Rosenbrock function

% Create a grid for visualization
[X, Y] = meshgrid(sort(x), sort(y)); % Create a meshgrid from sorted values

% Compute function values over the grid
ff = f1(X, Y);
figure()
% Plot the function surface
surf(X, Y, ff, 'edgecolor', 'none') % 3D surface plot with no edges

% Improve visualization with color mapping and shading
colormap jet % Use the "jet" colormap for better visualization
shading interp % Smooth the surface shading for better appearance
%%
% Mishra's Bird Function:
% This function is a non-convex optimization test function used for benchmarking.
% It has multiple local minima and a global minimum at approximately (-3.13, -1.57).

% Mishra's Bird function formula:
% f2(x, y) = sin(y) * exp((1 - cos(x))^2) + cos(x) * exp((1 - sin(y))^2) + (x - y)^2
f2 = @(x, y) sin(y) .* exp((1 - cos(x)).^2) + cos(x) .* exp((1 - sin(y)).^2) + (x - y).^2;

% Extract X and Y values from the first two columns of X_class1
x = X_class1(:,1); % X-coordinates
y = X_class1(:,2); % Y-coordinates

% Compute function values and store them in the fourth column of X_class1
X_class1(:,4) = f2(x, y); % Evaluate Mishra's Bird function

% Extract X and Y values from the first two columns of X_class2
x = X_class2(:,1); % X-coordinates
y = X_class2(:,2); % Y-coordinates

% Compute function values and store them in the fourth column of X_class2
X_class2(:,4) = f2(x, y); % Evaluate Mishra's Bird function

% Create a grid for visualization
[X, Y] = meshgrid(sort(x), sort(y)); % Create a meshgrid from sorted values

% Compute function values over the grid
ff = f2(X, Y);
figure()
% Plot the function surface
surf(X, Y, ff, 'edgecolor', 'none') % 3D surface plot with no edges

% Improve visualization with color mapping and shading
colormap jet % Use the "jet" colormap for better visualization
shading interp % Smooth the surface shading for better appearance

% Set axis limits for better visualization
xlim([15.00 25.73]) % X-axis range
ylim([-2.3 20.9])   % Y-axis range
zlim([-200 1000])   % Z-axis range

% Adjust the viewing angle for better perception of the function surface
view([-82 34]) % Set 3D view angle
%%
% Townsend Function (Modified):
% This function is a modified version of the Townsend function, commonly used in optimization problems.
% It includes non-linearity and oscillatory behavior due to trigonometric components.

% Modified Townsend function formula:
% f3(x, y) = - (cos(x - 0.1) * y)^2 - x * sin(3x + y)
f3 = @(x, y) - (cos(x - 0.1) .* y).^2 - x .* sin(3*x + y);

% Extract X and Y values from the first two columns of X_class1
x = X_class1(:,1); % X-coordinates
y = X_class1(:,2); % Y-coordinates

% Compute function values and store them in the fifth column of X_class1
X_class1(:,5) = f3(x, y); % Evaluate modified Townsend function

% Extract X and Y values from the first two columns of X_class2
x = X_class2(:,1); % X-coordinates
y = X_class2(:,2); % Y-coordinates

% Compute function values and store them in the fifth column of X_class2
X_class2(:,5) = f3(x, y); % Evaluate modified Townsend function

% Create a grid for visualization
[X, Y] = meshgrid(sort(x), sort(y)); % Create a meshgrid from sorted values

% Compute function values over the grid
ff = f3(X, Y);
figure()
% Plot the function surface
surf(X, Y, ff, 'edgecolor', 'none') % 3D surface plot with no edges

% Improve visualization with color mapping and shading
colormap jet % Use the "jet" colormap for better visualization
shading interp % Smooth the surface shading for better appearance

% Adjust the viewing angle for better perception of the function surface
view([-77 46]) % Set 3D view angle
%%
% Gomez and Levy Function (Modified):
% This function is a polynomial function used for optimization testing.
% It consists of quadratic, quartic, and sextic terms, making it highly nonlinear.

% Modified Gomez and Levy function formula:
% f4(x, y) = 4x^2 - 2.1x^4 + (1/3)x^6 + xy - 4y^2 + 4y^4
f4 = @(x, y) (4*x.^2) - (2.1*x.^4) + (1/3*x.^6) + (x.*y) - (4*y.^2) + (4*y.^4);

% Extract X and Y values from the first two columns of X_class1
x = X_class1(:,1); % X-coordinates
y = X_class1(:,2); % Y-coordinates

% Compute function values and store them in the sixth column of X_class1
X_class1(:,6) = f4(x, y); % Evaluate the modified Gomez and Levy function

% Extract X and Y values from the first two columns of X_class2
x = X_class2(:,1); % X-coordinates
y = X_class2(:,2); % Y-coordinates

% Compute function values and store them in the sixth column of X_class2
X_class2(:,6) = f4(x, y); % Evaluate the modified Gomez and Levy function

% Create a grid for visualization
[X, Y] = meshgrid(sort(x), sort(y)); % Create a meshgrid from sorted values

% Compute function values over the grid
ff = f4(X, Y);
figure()
% Plot the function surface
surf(X, Y, ff, 'edgecolor', 'none') % 3D surface plot with no edges

% Improve visualization with color mapping and shading
colormap jet % Use the "jet" colormap for better visualization
shading interp % Smooth the surface shading for better appearance
%%
% Simionescu Function:
% This function is a simple nonlinear function often used in constrained optimization problems.
% It has a linear multiplication term and a small scaling factor.

% Simionescu function formula:
% f5(x, y) = 0.1 * (x * y)
f5 = @(x, y) 0.1 * (x .* y);

% Extract X and Y values from the first two columns of X_class1
x = X_class1(:,1); % X-coordinates
y = X_class1(:,2); % Y-coordinates

% Compute function values and store them in the seventh column of X_class1
X_class1(:,7) = f5(x, y); % Evaluate Simionescu function

% Extract X and Y values from the first two columns of X_class2
x = X_class2(:,1); % X-coordinates
y = X_class2(:,2); % Y-coordinates

% Compute function values and store them in the seventh column of X_class2
X_class2(:,7) = f5(x, y); % Evaluate Simionescu function

% Create a grid for visualization
[X, Y] = meshgrid(sort(x), sort(y)); % Create a meshgrid from sorted values

% Compute function values over the grid
ff = f5(X, Y);
figure()
% Plot the function surface
surf(X, Y, ff, 'edgecolor', 'none') % 3D surface plot with no edges

% Improve visualization with color mapping and shading
colormap jet % Use the "jet" colormap for better visualization
shading interp % Smooth the surface shading for better appearance
%%
% Booth Function:
% The Booth function is a well-known test function for optimization problems.
% It has a global minimum at (1,3) where f(1,3) = 0.
% The function is continuous, convex, and differentiable, making it ideal for gradient-based optimization methods.

% Booth function formula:
% f6(x, y) = (x + 2*x - 7)^2 + (2*x + y - 5)^2
f6 = @(x, y) (x + 2*x - 7).^2 + (2*x + y - 5).^2;

% Extract X and Y values from the first two columns of X_class1
x = X_class1(:,1); % X-coordinates
y = X_class1(:,2); % Y-coordinates

% Compute function values and store them in the eighth column of X_class1
X_class1(:,8) = f6(x, y); % Evaluate Booth function

% Extract X and Y values from the first two columns of X_class2
x = X_class2(:,1); % X-coordinates
y = X_class2(:,2); % Y-coordinates

% Compute function values and store them in the eighth column of X_class2
X_class2(:,8) = f6(x, y); % Evaluate Booth function

% Create a grid for visualization
[X, Y] = meshgrid(sort(x), sort(y)); % Create a meshgrid from sorted values

% Compute function values over the grid
ff = f6(X, Y);
figure()
% Plot the function surface
surf(X, Y, ff, 'edgecolor', 'none') % 3D surface plot with no edges

% Improve visualization with color mapping and shading
colormap jet % Use the "jet" colormap for better visualization
shading interp % Smooth the surface shading for better appearance
%%
% Assign class labels to the datasets
% Class 1 is labeled as "1", and Class 2 is labeled as "2"
X_class1(:,end+1) = 1; % Append label column for class 1
X_class2(:,end+1) = 2; % Append label column for class 2

% -------------------------------------------------------------------------
% Splitting Data for Class 1
% -------------------------------------------------------------------------

% Split class 1 data into training (50%) and adaptation/test (50%)
c1 = cvpartition(X_class1(:,end), 'Holdout', 0.50); % 50% holdout for test/adaptation
set1 = training(c1); % Get logical index for training samples
indx_class1_train = find(set1 == 1); % Indices for training data
indx_class1_Adaption_test = find(set1 == 0); % Indices for adaptation/test data

% Create training and adaptation/test datasets for class 1
X_class1_train = X_class1(indx_class1_train, :); % Training data
X_class1_Adaption_test = X_class1(indx_class1_Adaption_test, :); % Adaptation/Test data

% Further split adaptation/test data into adaptation (30%) and test (20%)
c11 = cvpartition(X_class1_Adaption_test(:,end), 'Holdout', 0.30); % 30% holdout for test
set11 = training(c11); % Get logical index for adaptation samples
indx_class1_Adaption = find(set11 == 1); % Indices for adaptation data
indx_class1_test = find(set11 == 0); % Indices for test data

% Create adaptation and test datasets for class 1
X_class1_Adaption = X_class1_Adaption_test(indx_class1_Adaption, :); % Adaptation data
X_class1_test = X_class1_Adaption_test(indx_class1_test, :); % Test data

% -------------------------------------------------------------------------
% Splitting Data for Class 2
% -------------------------------------------------------------------------

% Split class 2 data into training (50%) and adaptation/test (50%)
c2 = cvpartition(X_class2(:,end), 'Holdout', 0.50); % 50% holdout for test/adaptation
set2 = training(c2); % Get logical index for training samples
indx_class2_train = find(set2 == 1); % Indices for training data
indx_class2_Adaption_test = find(set2 == 0); % Indices for adaptation/test data

% Create training and adaptation/test datasets for class 2
X_class2_train = X_class2(indx_class2_train, :); % Training data
X_class2_Adaption_test = X_class2(indx_class2_Adaption_test, :); % Adaptation/Test data

% Further split adaptation/test data into adaptation (30%) and test (20%)
c22 = cvpartition(X_class2_Adaption_test(:,end), 'Holdout', 0.30); % 30% holdout for test
set22 = training(c22); % Get logical index for adaptation samples
indx_class2_Adaption = find(set22 == 1); % Indices for adaptation data
indx_class2_test = find(set22 == 0); % Indices for test data

% Create adaptation and test datasets for class 2
X_class2_Adaption = X_class2_Adaption_test(indx_class2_Adaption, :); % Adaptation data
X_class2_test = X_class2_Adaption_test(indx_class2_test, :); % Test data
%%
% -------------------------------------------------------------------------
% Create and Export Training Data Tables
% -------------------------------------------------------------------------

% Convert Class 1 training data to table format
T1_train = array2table(X_class1_train, ...
    'VariableNames', {'X1','X2','Rosenbrock','Mishra','Townsend','Gomez','Simionescu','Booth','label'});

% Convert Class 2 training data to table format
T2_train = array2table(X_class2_train, ...
    'VariableNames', {'X1','X2','Rosenbrock','Mishra','Townsend','Gomez','Simionescu','Booth','label'});

% Combine training data from both classes into a single table
Data_train = [T1_train; T2_train];

% Create separate datasets for each function with X1, X2, function value, and label
Data_out1_train = Data_train(:, ["X1", "X2", "Rosenbrock", "label"]);
Data_out2_train = Data_train(:, ["X1", "X2", "Mishra", "label"]);
Data_out3_train = Data_train(:, ["X1", "X2", "Townsend", "label"]);
Data_out4_train = Data_train(:, ["X1", "X2", "Gomez", "label"]);
Data_out5_train = Data_train(:, ["X1", "X2", "Simionescu", "label"]);
Data_out6_train = Data_train(:, ["X1", "X2", "Booth", "label"]);

% -------------------------------------------------------------------------
% Create and Export Adaptation Data Tables
% -------------------------------------------------------------------------

% Convert Class 1 adaptation data to table format
T1_adaption = array2table(X_class1_Adaption, ...
    'VariableNames', {'X1','X2','Rosenbrock','Mishra','Townsend','Gomez','Simionescu','Booth','label'});

% Convert Class 2 adaptation data to table format
T2_adaption = array2table(X_class2_Adaption, ...
    'VariableNames', {'X1','X2','Rosenbrock','Mishra','Townsend','Gomez','Simionescu','Booth','label'});

% Combine adaptation data from both classes into a single table
Data_Adaption = [T1_adaption; T2_adaption];

% Create separate adaptation datasets for each function
Data_out1_Adaption = Data_Adaption(:, ["X1", "X2", "Rosenbrock", "label"]);
Data_out2_Adaption = Data_Adaption(:, ["X1", "X2", "Mishra", "label"]);
Data_out3_Adaption = Data_Adaption(:, ["X1", "X2", "Townsend", "label"]);
Data_out4_Adaption = Data_Adaption(:, ["X1", "X2", "Gomez", "label"]);
Data_out5_Adaption = Data_Adaption(:, ["X1", "X2", "Simionescu", "label"]);
Data_out6_Adaption = Data_Adaption(:, ["X1", "X2", "Booth", "label"]);

% -------------------------------------------------------------------------
% Create and Export Test Data Tables
% -------------------------------------------------------------------------

% Convert Class 1 test data to table format
T1_test = array2table(X_class1_test, ...
    'VariableNames', {'X1','X2','Rosenbrock','Mishra','Townsend','Gomez','Simionescu','Booth','label'});

% Convert Class 2 test data to table format
T2_test = array2table(X_class2_test, ...
    'VariableNames', {'X1','X2','Rosenbrock','Mishra','Townsend','Gomez','Simionescu','Booth','label'});

% Combine test data from both classes into a single table
Data_test = [T1_test; T2_test];

% Create separate test datasets for each function
Data_out1_test = Data_test(:, ["X1", "X2", "Rosenbrock", "label"]);
Data_out2_test = Data_test(:, ["X1", "X2", "Mishra", "label"]);
Data_out3_test = Data_test(:, ["X1", "X2", "Townsend", "label"]);
Data_out4_test = Data_test(:, ["X1", "X2", "Gomez", "label"]);
Data_out5_test = Data_test(:, ["X1", "X2", "Simionescu", "label"]);
Data_out6_test = Data_test(:, ["X1", "X2", "Booth", "label"]);

%%
% -------------------------------------------------------------------------
% Delete Old Excel Files Before Exporting New Data
% This ensures that the dataset is refreshed without keeping outdated files
% -------------------------------------------------------------------------

% Delete old training data files
delete Data_out1_train.xlsx  % Delete previous Rosenbrock training data
delete Data_out2_train.xlsx  % Delete previous Mishra training data
delete Data_out3_train.xlsx  % Delete previous Townsend training data
delete Data_out4_train.xlsx  % Delete previous Gomez training data
delete Data_out5_train.xlsx  % Delete previous Simionescu training data
delete Data_out6_train.xlsx  % Delete previous Booth training data

% Delete old adaptation data files
delete Data_out1_Adaption.xlsx  % Delete previous Rosenbrock adaptation data
delete Data_out2_Adaption.xlsx  % Delete previous Mishra adaptation data
delete Data_out3_Adaption.xlsx  % Delete previous Townsend adaptation data
delete Data_out4_Adaption.xlsx  % Delete previous Gomez adaptation data
delete Data_out5_Adaption.xlsx  % Delete previous Simionescu adaptation data
delete Data_out6_Adaption.xlsx  % Delete previous Booth adaptation data

% Delete old test data files
delete Data_out1_test.xlsx  % Delete previous Rosenbrock test data
delete Data_out2_test.xlsx  % Delete previous Mishra test data
delete Data_out3_test.xlsx  % Delete previous Townsend test data
delete Data_out4_test.xlsx  % Delete previous Gomez test data
delete Data_out5_test.xlsx  % Delete previous Simionescu test data
delete Data_out6_test.xlsx  % Delete previous Booth test data
%%
% -------------------------------------------------------------------------
% Export Training Data to Excel Files
% This saves each function's output separately in individual Excel files
% -------------------------------------------------------------------------

filename = 'Data_out1_train.xlsx';
writetable(Data_out1_train, filename, 'Sheet', 1); % Write Rosenbrock function training data

filename = 'Data_out2_train.xlsx';
writetable(Data_out2_train, filename, 'Sheet', 1); % Write Mishra function training data

filename = 'Data_out3_train.xlsx';
writetable(Data_out3_train, filename, 'Sheet', 1); % Write Townsend function training data

filename = 'Data_out4_train.xlsx';
writetable(Data_out4_train, filename, 'Sheet', 1); % Write Gomez function training data

filename = 'Data_out5_train.xlsx';
writetable(Data_out5_train, filename, 'Sheet', 1); % Write Simionescu function training data

filename = 'Data_out6_train.xlsx';
writetable(Data_out6_train, filename, 'Sheet', 1); % Write Booth function training data

% -------------------------------------------------------------------------
% Export Adaptation Data to Excel Files
% This saves adaptation datasets in separate Excel files
% -------------------------------------------------------------------------

filename = 'Data_out1_Adaption.xlsx';
writetable(Data_out1_Adaption, filename, 'Sheet', 1); % Write Rosenbrock adaptation data

filename = 'Data_out2_Adaption.xlsx';
writetable(Data_out2_Adaption, filename, 'Sheet', 1); % Write Mishra adaptation data

filename = 'Data_out3_Adaption.xlsx';
writetable(Data_out3_Adaption, filename, 'Sheet', 1); % Write Townsend adaptation data

filename = 'Data_out4_Adaption.xlsx';
writetable(Data_out4_Adaption, filename, 'Sheet', 1); % Write Gomez adaptation data

filename = 'Data_out5_Adaption.xlsx';
writetable(Data_out5_Adaption, filename, 'Sheet', 1); % Write Simionescu adaptation data

filename = 'Data_out6_Adaption.xlsx';
writetable(Data_out6_Adaption, filename, 'Sheet', 1); % Write Booth adaptation data

% -------------------------------------------------------------------------
% Export Test Data to Excel Files
% This saves test datasets in separate Excel files
% -------------------------------------------------------------------------

filename = 'Data_out1_test.xlsx';
writetable(Data_out1_test, filename, 'Sheet', 1); % Write Rosenbrock test data

filename = 'Data_out2_test.xlsx';
writetable(Data_out2_test, filename, 'Sheet', 1); % Write Mishra test data

filename = 'Data_out3_test.xlsx';
writetable(Data_out3_test, filename, 'Sheet', 1); % Write Townsend test data

filename = 'Data_out4_test.xlsx';
writetable(Data_out4_test, filename, 'Sheet', 1); % Write Gomez test data

filename = 'Data_out5_test.xlsx';
writetable(Data_out5_test, filename, 'Sheet', 1); % Write Simionescu test data

filename = 'Data_out6_test.xlsx';
writetable(Data_out6_test, filename, 'Sheet', 1); % Write Booth test data







