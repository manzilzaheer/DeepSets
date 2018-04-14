function [X, Y, X_parameter] = generate_task1_dataset(L, N)
%Generates input-output data for supervised entropy learning on 2D Gaussian.
%
%OUTPUT:
%   X: X{k} = samples from the k^th bag; one column of X{k} is one sample.
%   Y: Y(k) = real label of the k^th bag.
%   X_parameter: vector of rotation angles. 

% fix seed for same parameters
    rng(3);

%initialization:
    X = cell(L, 1);  %inputs (bags representing the input distributions)
    Y = zeros(L, 1); %output labels
    
%X_parameter:
    d = 2; %dimenson
    A = rand(d); %randomly initialize cholesky of sigma
    X_parameter = linspace(0, pi, L)'; %rotation angles
    
%X,Y:
    rng('shuffle');
    for nL = 1 : L
        %A_nL:
            A_nL = rotation_matrix(X_parameter(nL)) * A;
        %X:    
            X{nL} = (A_nL * randn(d, N))'; %bag representing a rotated normal distribution
        %Y (entropy of the first coordinate):
            M = A_nL * A_nL.'; 
            s = M(1, 1);
            Y(nL) = 1/2 * log(2*pi*exp(1)*s^2);
    end

%plot the generated dataset:
    figure;
    plot(X_parameter, Y); xlabel('Rotation angle'); ylabel('Entropy of the first coordinate');
    
    
function [R] = rotation_matrix(angle)
    C = cos(angle);
    S = sin(angle);
    R = [C, -S; S, C];