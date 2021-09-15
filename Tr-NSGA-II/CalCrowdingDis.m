function f = CalCrowdingDis(x, M, V)

% [temp,index_of_fronts] = sort(x(:,M + V + 1));
% 
% for i = 1 : length(index_of_fronts)
%     sorted_based_on_front(i,:) = x(index_of_fronts(i),:);
% end
current_index = 0;

[lengthFfront, m] = size(x);

clear m

%    objective = [];
distance = 0;
y = x(:,1:M+V+1);
previous_index = current_index + 1;
% for i = 1 : lengthFfront
%     y(i,:) = sorted_based_on_front(current_index + i,:);
% end
current_index = lengthFfront;
% Sort each individual based on the objective
sorted_based_on_objective = [];
for i = 1 : M
%     index_of_objectives = size(y(:,V + i),1);
    [sorted_based_on_objective, index_of_objectives] = ...
        sort(y(:,V + i));
    sorted_based_on_objective = [];
    for j = 1 : length(index_of_objectives)
        sorted_based_on_objective(j,:) = y(index_of_objectives(j),:);
    end
    
    f_max = ...
        sorted_based_on_objective(length(index_of_objectives), V + i);
    f_min = sorted_based_on_objective(1, V + i);
    y(index_of_objectives(length(index_of_objectives)),M + V + 1 + i)...
        = Inf;
    y(index_of_objectives(1),M + V + 1 + i) = Inf;
     for j = 2 : length(index_of_objectives) - 1
        next_obj  = sorted_based_on_objective(j + 1,V + i);
        previous_obj  = sorted_based_on_objective(j - 1,V + i);
        if (f_max - f_min == 0)
            y(index_of_objectives(j),M + V + 1 + i) = Inf;
        else
            y(index_of_objectives(j),M + V + 1 + i) = ...
                 (next_obj - previous_obj)/(f_max - f_min);
        end
     end
end
distance = [];
distance(:,1) = zeros(lengthFfront,1);
for i = 1 : M
    distance(:,1) = distance(:,1) + y(:,M + V + 1 + i);
end
y(:,M + V + 2) = distance;
y = y(:,1 : M + V + 2);
z(previous_index:current_index,:) = y;

for i = 1:current_index
    density = 0;
    for j = 1:current_index
        dis = 0;
        for obj = 1:M
            dis = dis + (z(i,obj) - z(j,obj))*(z(i,obj) - z(j,obj));
        end
        if(sqrt(dis) < 1.0E-10)
            dis = 1.0E10;
        else
            dis = 1/sqrt(dis);
        end
        density = dis + density;
    end
    z(i,M+V+3) = density;
end

f = z();