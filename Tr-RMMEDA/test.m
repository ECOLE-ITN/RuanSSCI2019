for group = 1:1:1
    for T = 1:1:1
        filename1 = ['E:\All codes of TR-DMOEA\TR-DMOEA\Tr-RMMEDA\Results\POF-AfTr\FDA4\group' num2str(group) '\POF-AfTr' num2str(T) '.txt'];
        fprintf('%s',filename1);
        data = importdata(filename1);
        A = mean(data);
        fprintf('%d\n',A);
    end
end
% data=importdata('E:\ด๚ย๋\TR-DMOEA\สตั้\Tr-RMMEDA\FDA5_iso\group1\iterIGD-T1.txt');
% A = mean(data);