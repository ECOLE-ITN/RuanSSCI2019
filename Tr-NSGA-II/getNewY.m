function Y=getNewY(Xs, Xt, X, W, kind, p1, p2, p3)
%��������X����ӳ��(X����ΪԴ���Ŀ�����������)
%Xs��Դ������
%Xt��Ŀ�������ݣ���Xs������ͬ
%X�����任��������Xs������ͬ
%W���任����n1+n2->k
%kind���˺���ѡ��:'Gaussian'��'Laplacian'��'Polynomial',����һ�ɷ���-1
%p1,p2,p3���˺�����Ҫ�����Ĳ���

    n1 = size(Xs, 2);
	n2 = size(Xt, 2);
	n3 = size(X, 2);
    
    for j=1:n3
        for i=1:n1 
            K(i,j)=getKernel(Xs(:,i), X(:,j), kind, p1, p2, p3);
        end
        for i=1:n2
            K(i+n1,j)=getKernel(Xt(:,i), X(:,j), kind, p1, p2, p3);
        end
    end
    
    Y=W'*K;