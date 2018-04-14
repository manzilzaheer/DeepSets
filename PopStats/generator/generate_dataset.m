function [] = generate_dataset()
    clear; %clear memory

    for i = 1:4
        task = ['generate_task', num2str(i), '_dataset'];
        generator = str2func(task);
        out_directory = ['data/task', num2str(i), '/'] %['./data/task', num2str(i), '/']; % 
        disp(task);
        disp(out_directory);

        L=512; %Number of sets in validation/test
        N=512; %Number of points per set

        %Generate test dataset
        save_dataset(L, N, [out_directory,'test.mat'], generator)

        %Generate validation dataset
        save_dataset(L, N, [out_directory,'val.mat'], generator)

        %Generate truth for plotting
        save_dataset(2^14, 1, [out_directory,'truth.mat'], generator)

        %create dataset of different size:
        for logL = 7:17 %number of train distributions
            L = 2^logL;
            disp(L);
            save_dataset(L, N, [out_directory, 'data_', int2str(logL), '.mat'], generator)
        end
        display(['Data generated for task' num2str(i)])
    end
    display('Done')
end


function [] = save_dataset(L, N, fname, generator)
    [X, Y, X_parameter] = generator(L, N);
    X = cell2mat(X);
    save(fname, '-v7.3')
end
