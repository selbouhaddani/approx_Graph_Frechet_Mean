%Displaying the Frechet mean and an element of the sample set of graphs

A = sampleAdjSet{1};
A = A-1;
A = -1*A;
deg = sum(A);

[sorted,I] = sort(deg);
sortedA = A(I,I);
p_i = sort(p_i, 'descend');
frechetMean = rand_adj(p_i,q_i,n);
frechetMean = frechetMean-1;
frechetMean = -1*(frechetMean);
degMean = sum(frechetMean);

[sortedDegMean,I_mean] = sort(degMean);

sortedfrechetMean = frechetMean(I_mean,I_mean);

figure(1)
hold on
subplot(1,2,1)
imagesc(A)
xlabel('An Observed Graph','FontSize', 20)
set(gca,'XTick',[]);
set(gca,'YTick',[]);
subplot(1,2,2)
imagesc(frechetMean)
xlabel('Approximate Empirical Frechet Mean: G_{SBM}^*','FontSize', 20)
cmap = gray(256);
cmap(9,:) = [1,0,0];
set(gca,'XTick',[]);
set(gca,'YTick',[]);
colormap(cmap);

function [A] = rand_adj(p,q,n)
    %c is the num of communities
    [numCom,~] = size(p);
    largestCom = floor(n/numCom) + rem(n,numCom);
    otherCom = floor(n/numCom);
    
    comSizes = otherCom*ones(1,numCom);
    comSizes(1) = largestCom;
    
    c = cell(numel(comSizes),1);
    c_temp = cell(numel(comSizes),1);
    for i = 1:numel(comSizes)
        c{i} = binornd(1,p(i)*ones(comSizes(i)));
        c_temp{i} = c{i} - 0.5;
    end
    
    A_temp = blkdiag(c_temp{:});
    
    A = blkdiag(c{:});
    B = zeros(n,n);
    for i = 1:n
        for j = i:n
            if A_temp(i,j) == 0
                B(i,j) = binornd(1,q);
            end
        end
    end
    A = A + B;
    A = triu(A);
    A = A + A';
    A = A - diag(diag(A));

end

