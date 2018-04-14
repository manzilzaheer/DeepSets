function [X, Y, X_parameter] = generate_task5_dataset(L, N)
%Generates input-output data for supervised mutual information learning on rank-1 update of a Gaussian matrix
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
    d = 32; %dimenson
    %v = rand(d,1); %randomly initialize diagonal of sigma
    %v = v./norm(v);
    A = eye(d); % diag(2*v+0.5);
    v = rand(d,1);
    v = v./norm(v);
    X_parameter = linspace(0,1,L)';
    
%X,Y:
    rng('shuffle');
    for nL = 1 : L
        %A_nL:
            A_nL = cholupdate(A, sqrt(X_parameter(nL))*v);
        %X:    
            X{nL} = (A_nL * randn(d, N))'; %bag representing a rotated normal distribution
        %Y (Mutual information):
            Y(nL) = -sum(log(diag(A_nL))) + 0.5*sum(log(sum(A_nL.^2,2)));
    end

%plot the generated dataset:
    figure;
    plot(X_parameter, Y); xlabel('Rotation angle'); ylabel('Entropy of the first coordinate');

end
    
function [L] = cholupdate(L,x)
    n = length(x);
    for k=1:n
        r = sqrt(L(k,k)^2 + x(k)^2);
        c = r / L(k, k);
        s = x(k) / L(k, k);
        L(k, k) = r;
        L(k+1:n,k) = (L(k+1:n,k) + s*x(k+1:n)) / c;
        x(k+1:n) = c*x(k+1:n) - s*L(k+1:n,k);
    end
end
