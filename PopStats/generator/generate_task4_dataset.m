function [X, Y, X_parameter] = generate_task4_dataset(L, N)
%Generates input-output data for supervised mutual information learning on 32D Gaussian.
%
%OUTPUT:
%   X: X{k} = samples from the k^th bag; one column of X{k} is one sample.
%   Y: Y(k) = real label of the k^th bag.
%   X_parameter: No significance, just sorted index. 

%initialization:
    X = cell(L, 1);  %inputs (bags representing the input distributions)
    Y = zeros(L, 1); %output labels
    
%X_parameter:
    d = 32;%dimenson
    X_parameter = linspace(0,1,L)';
    
%X,Y:
    rng('shuffle');
    for nL = 1 : L
        %A_nL:
            A = rand(d);
            A = (A*A' + eye(d))./sum(sum(A.^2));
            A_nL = chol(A);
        %X: 
            u = randn(d,N);
            X{nL} = (A_nL * u)'; %bags representing 32d normal distributions
        %Y (mutual information of multivariate gaussian):
            Y(nL) = -sum(log(diag(A_nL))) + 0.5*sum(log(sum(A_nL.^2,2)));
    end

%sort X,Y:
    [Y, I] = sort(Y);
    X = X(I);

%plot the generated dataset:
    figure;
    plot(X_parameter, Y); xlabel('Sorted Index'); ylabel('Mutual information');
