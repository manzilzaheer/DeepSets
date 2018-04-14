function [X, Y, X_parameter] = generate_task2_dataset(L, N)
%Generates input-output data for supervised mutual information learning on varing corelation of Gaussian matrix
%L must be a divisible by 4
%
%OUTPUT:
%   X: X{k} = samples from the k^th bag; one column of X{k} is one sample.
%   Y: Y(k) = real label of the k^th bag.
%   X_parameter: vector of correlation coefficient. 

%verification:
    if mod(L, 4) ~= 0
        L = L + 4 - mod(L, 4);
        disp(['WARNING: L not divisible 4, increasing it to next multiple of 4. New value of L = ' num2str(L)]);
    end

% fix seed for same parameters
    rng(3);

%initialization:
    X = cell(L,1);  %inputs (bags representing the input distributions)
    Y = zeros(L,1); %output labels    
    
%X_parameter:
    d = 16; %dimension

    X_parameter = [linspace(0.01,0.99,L/4), sqrt(1-exp(-4*linspace(0.01,0.99,L/4)))]; %correlation coefficients
    X_parameter = sort([-X_parameter,X_parameter])';

    A = rand(d/2);
    A = (A*A' + eye(d/2))./sum(sum(A.^2));

%X,Y:
    rng('shuffle');
    for nL = 1 : L
        %A_nL:
            A_t = [A, X_parameter(nL) * A; X_parameter(nL) * A, A];
            A_nL = chol(A_t);
        %X: 
            u = randn(d,N);
            X{nL} = (A_nL * u)'; %bag representing a normal distribution with correlation
        %Y (mutual info):
            Y(nL) =  -1/2 * log(1-X_parameter(nL)^2);
    end
               
%plot the generated dataset:
    figure;
    plot(X_parameter,Y); xlabel('Correlation coefficient'); ylabel('Mutual information');
    
