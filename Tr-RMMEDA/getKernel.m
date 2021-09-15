function k=getKernel(a, b, kind, p1, p2, p3)
%��˺���kernel(a,b)��ֵ
%a,b��double��������
%kind��'Gaussian'��'Laplacian'��'Polynomial',����һ�ɷ���-1
%k�����غ˵�ֵ

    if strcmp(kind,'Polynomial')
        k=dot(a,b);
        k=(p1*k+p2)^p3;
    elseif strcmp(kind,'Laplacian')
        a=a-b;
        k=a'*a;
        k=exp(-p1*sqrt(k));
    elseif strcmp(kind,'Gaussian')
        a=a-b;
        k=a'*a;
        k=exp(-p1*k);
    else
        k=-1;
    end