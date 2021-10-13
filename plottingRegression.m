[numDataPoints, ~] = size(plottingData);
sampleEigData1 = zeros(30,1);
sampleEigData2 = zeros(30,1);
sampleEigData3 = zeros(30,1);
for i = 1:30
    sampleEigData1(i) = sampleEigSet{i}(1);
    sampleEigData2(i) = sampleEigSet{i}(2);
    sampleEigData3(i) = sampleEigSet{i}(3); 
end

%Compute theoretical Eigenvalues of approximate Empirical Frechet Mean at
%each t
theoreticalApproxFMEigs1 = zeros(numDataPoints,1);
theoreticalApproxFMEigs2 = zeros(numDataPoints,1);
theoreticalApproxFMEigs3 = zeros(numDataPoints,1);

for i = 1:numDataPoints
    approxEigs_t = eigsApprox(plottingData{i,2},plottingData{i,3},n);
    theoreticalApproxFMEigs1(i) = approxEigs_t(1);
    theoreticalApproxFMEigs2(i) = approxEigs_t(2);
    theoreticalApproxFMEigs3(i) = approxEigs_t(3);
end

%Estimate approximate empirical Frechet mean graph and compute the 3
%largest eigenvalues


approxEmpFMEigs = zeros(3,numDataPoints);
T = 1; %How many samples to draw from SBM to estimate G_N^*
for k = 1:numDataPoints
    frechetMeanEigs = [];
    for i = 1:T
        frechetMeanEigs = [frechetMeanEigs, sort(eig(rand_adj(plottingData{k,2},plottingData{k,3},n)),'descend')];
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
    
    approxEmpFMEigs(1,k) = approxMeanEigs(1);
    approxEmpFMEigs(2,k) = approxMeanEigs(2); 
    approxEmpFMEigs(3,k) = approxMeanEigs(3);
end
t = [sort(t_samples);sort(t_samples);sort(t_samples)];

sampleEigs = [sampleEigData1; sampleEigData2; sampleEigData3];

figure(2)
plot(t,sampleEigs,'o','color','black','markersize',10)
hold on
plot([0:stepSize:1,0:stepSize:1,0:stepSize:1],[approxEmpFMEigs(1,:),approxEmpFMEigs(2,:),approxEmpFMEigs(3,:)],'s','color','blue','markersize',10)
plot([0:stepSize:1,0:stepSize:1,0:stepSize:1],[theoreticalApproxFMEigs1;theoreticalApproxFMEigs2;theoreticalApproxFMEigs3],'x','color','red','markersize',10)

legend('Sampled Eigenvalues','Estimated Eigenvalues', 'Theoretical Eigenvalues')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Estimate eigs of Frechet mean given p,q,s
function [eigs_i] = eigsApprox(p_i,q_i,n)
    [c,~] = size(p_i);

    %compute the expected adj matrix
    EA = exp_rand_adj(p_i,q_i,n);
    [EV,ED] = eigs(EA,c);
    eigs_i = diag(ED(1:c,1:c));
end

%compute a random adj matrix from a SBM
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

%Compute the expected adj matrix of a SBM
function [A] = exp_rand_adj(p,q,n)
    [numCom,~] = size(p);
    
    largestCom = floor(n/numCom) + rem(n,numCom);
    otherCom = floor(n/numCom);
    
    comSizes = otherCom*ones(1,numCom);
    comSizes(1) = largestCom;
    
    c = cell(numel(comSizes),1);
    for i = 1:numel(comSizes)
        c{i} = (p(i)-q)*ones(comSizes(i),comSizes(i));
    end
    
    A = blkdiag(c{:})+q;
    A = A - diag(diag(A));
end