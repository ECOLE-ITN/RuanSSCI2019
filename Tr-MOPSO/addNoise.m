function newPopulation = addNoise(init_population, Nini, n)
    Num = size(init_population,1);
    newPopNumber = Nini - Num;
    newPopulation = zeros(newPopNumber,n);
    
    for i = 1:newPopNumber
        %����Щ������оֲ�����
        index = randperm(Num,1);
        
        newPopulation(i,:) = init_population(index,:);
        index2 = randperm(n,round(n*0.3));%ȡ�������߱�����30%��ͻ��

        temp = init_population(index,index2) + normrnd(0,0.5);
        
        upp = temp>1;
        loww = temp<0;
        temp(upp) = (init_population(index,upp)+1)/2;
        temp(loww) = (init_population(index,loww)-0)/2;
        %%%%%%%%%%%%%%%%2019.06.01
        for j = 1:1:size(index2,2)
            newPopulation(i,index2(1,j)) = temp(j);
        end
    end
end