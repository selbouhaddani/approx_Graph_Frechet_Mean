sampleEigs = [];
frechetMeanEigs = [];
for i = 1:N
    sampleEigs = [sampleEigs, eig(sampleAdjSet{i})];
end

avgEig = mean(sampleEigs');
avgEig = sort(avgEig,'descend');

T = 5;
for i = 1:T
    frechetMeanEigs = [frechetMeanEigs, eig(rand_adj(p_i,q_i,n))];
end

distMatrix = zeros(T,T);
for i = 1:T
    for j = 1:T
        distMatrix(i,j) = norm(frechetMeanEigs(i) - norm(frechetMeanEigs(j)));
    end
end

distMatSum = sum(distMatrix);
[entry,index] = min(distMatSum);
approxMeanEigs = frechetMeanEigs(:,index);

approxMeanEigs = sort(approxMeanEigs,'descend');

avgBulkEigs = avgEig(c_star+1:end);
approxMeanBulkEigs = approxMeanEigs(c_star+1:end);

figure(1)
subplot(1,2,1)
[f,xi] = ksdensity(avgBulkEigs);
hold on
plot(xi,f,'o','color','k')

[f2,xi2] = ksdensity(approxMeanBulkEigs);
plot(xi2,f2,'s','color','b')
ylim([0 max(1.5*f)])
legend('Average bulk eigenvalues from M_3', 'Bulk eigenvalues of G^*_{SBM}', 'FontSize', 18)
xlabel('Estimated pdf of eigenvalues (not normalized)', 'FontSize', 20)
ylabel('Probability', 'FontSize', 20)

extremeEigsApproxMean = sort(approxMeanEigs(1:c_star),'ascend');

error = abs(eigs_i - avgEigVal);
avgEigVal = sort(avgEigVal, 'ascend');
eigs_i = sort(eigs_i,'ascend');
subplot(1,2,2)
hold on
plot(1:c_star, sort(avgEigVal(1:c_star),'descend'),'o','color','black', 'markersize',10)
plot(1:c_star, sort(extremeEigsApproxMean(1:c_star),'descend'),'s','color','blue', 'markersize',10)
plot(1:c_star, sort(eigs_i,'descend'),'x','color','red', 'markersize',10)
ylim([0 max(1.5*eigs_i)])
legend('Average extreme eigenvalues from M_3', 'Estimated extreme eigenvalues of G_{SBM}^*', 'Theoretical extreme eigenvalues of G_{SBM}^*', 'FontSize', 18)
xlabel('Index of extreme eigenvalue', 'FontSize', 20)
ylabel('Value of extreme eigenvalue', 'FontSize', 20)

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

function [A] = ctsA(f,n)
    x = 0:1/(n-1):1;
    y = 0:1/(n-1):1;
    [X,Y] = meshgrid(x,y);
    
    EA = f(X,Y);
    
    A = binornd(1,EA,size(EA));
    U = triu(A,1);
    A = U + U';
    A = A - diag(diag(A));
    
end