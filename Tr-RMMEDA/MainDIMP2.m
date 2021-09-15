%%
% <html><h2>Main function for Tr-RMMEDA</h2></html>

clear
clc

functions = {'FDA4', 'FDA5', 'FDA5_iso', 'FDA5_dec', 'DIMP2', 'dMOP2', ...
     'dMOP2_iso', 'dMOP2_dec', 'dMOP3', 'HE2', 'HE7', 'HE9'};

for testfunc = 5
    clearvars -except functions testfunc
    
    %% step 1. Initialize objective functions
    T_paramiter = [
        10 5 100
        10 10 200
        10 25 500
        10 50 1000
        1 10 200
        1 50 1000
        20 10 200
        20 50 1000
        ];
    
     ns = [12 12 12 12 10 10 10 10 10 30 10 10];
    nobjs = [3 3 3 3 2 2 2 2 2 2 2 2];
    Problem.Name = functions{testfunc}; % name of test problem
    Problem.FObj = eval(['@testFunctions.' Problem.Name]);
    
    n = ns(testfunc);                   %number of X
    Problem.NObj = nobjs(testfunc);     % number of objectives
    
    Problem.XLow    = zeros(n,1);       % lower boundary of decision variables
    Problem.XUpp    = ones(n,1);        % upper boundary of decision variables
    [n, ~, Ft] = getFunc(functions{testfunc});
    Generator.Name  = 'LPCA';           % name of generator
    Generator.NClu  = 5;                % parameter of generator, the number of clusters(default)
    Generator.Iter  = 50;               % maximum trainning steps in LPCA
    Generator.Exte  = 0.25;             % parameter of generator, extension rate(default)
    
    NIni = 200;
    IterMax = 50;
    
    %% step 2. Iterate through enviroment parameters
    for group =1:size(T_paramiter,1)
        mkdir(['./Results/POF-AfTr/' functions{testfunc} '/group' num2str(group)]);
        mkdir(['./Results/IGD/' functions{testfunc} '/group' num2str(group)]);
        fprintf('running %s \n',functions{testfunc});
        
        t = 0;       %the initial moment
        TruePOF = [];
        
        %% step 3. use RM-MEDA to get a POF at the initial moment with randomly generated population
        Pareto = RMMEDA( Problem, Generator, NIni, IterMax, t, testfunc, group, 0, TruePOF);
        POF = Pareto.F';
        POS = Pareto.X';
        for T = 1:T_paramiter(group,3)/T_paramiter(group,2)
           %% step 4. use TCA to get the initial population at the next moment
         
            % Initialize random populations
            sampleN = 400; 
            Xs = rand(sampleN, n);
            for i=1:1:sampleN
                [Fs(:,i),~] = Problem.FObj(Xs(i,:),t);
            end
            t= 1/T_paramiter(group,1)*(T-1);    % next moment
            F = @(x)Ft(x, t);
            Xt = rand(sampleN, n);
            
            for i=1:1:sampleN
                [Fa(:,i),~] = Problem.FObj(Xt(i,:),t);
            end
            len_POS = size(POS,1);
            POF_BeTr = zeros(Problem.NObj,len_POS);
            for i=1:1:len_POS
                [POF_BeTr(:,i),~] = Problem.FObj(POS(i,:),t);
            end
            POF_temp1 = zeros(Problem.NObj,len_POS);
            POF_temp1 = POF_BeTr';
            len_POF_BeTr = size(POF_BeTr,2);
           
            % Find the latent space of domain adaptation
            mu = 0.5;
            lambda = 'unused';
            dim = 20;
            % replace the Gaussian kernel with a linear one
            kind = 'Polynomial';
            p1 = 1;
            p2 = 0;
            p3 = 1;
            W = getW(Fs, Fa, mu, lambda, dim, kind, p1, p2, p3);
            POF_deduced = getNewY(Fs, Fa, POF', W, kind, p1, p2, p3);
            
            % Get initial population by POF_deduced
            dis_px = @(p, x)sum((getNewY(Fs, Fa, F(x)', W, kind, p1, p2, p3) - p).^2);
            initn = size(POF_deduced, 2);
            init_population = zeros(initn, n);
            init_temp = zeros(initn, n);
            for i = 1:initn
                individual = fmincon(@(x)dis_px(POF_deduced(:,i), x), rand(1,n), ...
                     [], [], [], [], zeros(1,n), ones(1,n), [], optimset('display', 'off'));
                minimization(i) = sum((getNewY(Fs, Fa, F(individual)', W, kind, p1, p2, p3) - POF_deduced(:,i)).^2);
                init_temp(i,:) = individual;
            end
            
            %Evaluate the transferred solutions
            POF_AfTr = zeros(Problem.NObj,len_POS);
            for i=1:1:initn
                [POF_AfTr(:,i),~] = Problem.FObj(init_temp(i,:),t);
            end
            POF_temp2 = zeros(Problem.NObj,len_POS);
            POF_temp2 = POF_AfTr';  % POF_temp2 = [N M]  POF_AfTr = [M N]
            
            % combine transferred and copied solutions, then select a new
            % population as the initial one.
            chromosome = zeros(initn+len_POS, Problem.NObj + n);
            leng1 = size(chromosome,2);
            chromosome(1:initn,1:n) = POS;
            chromosome(1:initn,n+1:n+Problem.NObj) = POF_temp1;
            chromosome(initn+1:2*initn,1:n) = init_temp;
            chromosome(initn+1:2*initn,n+1:n+Problem.NObj) = POF_temp2;

            chromosome = non_domination_sort_mod(chromosome,Problem.NObj,n);
            temp_chromosome = replace_chromosome(chromosome, Problem.NObj, n, initn);
            init_population = temp_chromosome(1:initn,1:n);
            
            len_POF_AfTr = size(POF_AfTr,2);
            
           %% Output POF after transfer
            FILEPATH_PFAT = ['./Results/POF-AfTr/' functions{testfunc} '/group' num2str(group) '/'];
            filename_PFAT = ['POF-AfTr' num2str(T) '.txt'];
            save([FILEPATH_PFAT,filename_PFAT],'POF_temp2','-ascii');
            
           %% step 5. use RM-MEDA to get the POF at every moment with the initial population
            TruePOF = getBenchmarkPOF(testfunc,group,T);
            %fprintf('running %d, %d, %d',NIni,initn,size(init_population,1));
            % IterMax = T_paramiter(group,2);
            Pareto = RMMEDA( Problem, Generator, NIni, T_paramiter(group,2), t, testfunc, group, T, TruePOF, init_population);
            POF = Pareto.F';
            POS = Pareto.X';
            %% Calculate the IGD of before and after transfer
            addpath(genpath('../Metrics'));%%%%
            IGD_AfTr(T) = IGD(POF_temp2,TruePOF);
        end
        FILEPATH_IGDAf = ['./Results/POF-AfTr/' functions{testfunc} '/group' num2str(group) '/'];
        filename_IGDAf = ['IGD_AfTr'  '.txt'];
        save([FILEPATH_IGDAf,filename_IGDAf],'IGD_AfTr','-ascii');
    end
end