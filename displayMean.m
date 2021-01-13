%Displaying the Frechet mean and an element of the sample set of graphs

A = sampleAdjSet{1};
A = A-1;
A = -1*A;
deg = sum(A);

[sorted,I] = sort(deg);
sortedA = A(I,I);

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
xlabel('An Observed Graph')
subplot(1,2,2)
imagesc(frechetMean)
xlabel('The Frechet Mean')
cmap = gray(256);
cmap(9,:) = [1,0,0];
colormap(cmap);

function [A] = rand_adj(p,q,n)
    %c is the num of communities
    numCom = length(p);
    if numCom > 2
        qvec = q*ones(numCom*(numCom-1)/2,1);
        comSize = n/numCom;

        comGraphs = cell(numCom,1);

        for i = 1:numCom
            comGraphs{i} = random('bino', 1, p(i), comSize, comSize);
        end

        numInterCom = length(qvec);
        interComGraphs = cell(numInterCom,1);

        for i = 1:numInterCom
            interComGraphs{i} = random('bino', 1, qvec(i), comSize, comSize);
        end

        blockRows = cell(numCom,1);

        for i = 1:numCom
            blockRows{i} = [];
            for j = 1:numCom
                if j == i
                    blockRows{i} = [blockRows{i}, comGraphs{j}];
                else
                    blockRows{i} = [blockRows{i}, interComGraphs{j}];
                end
            end
        end

        adjMat = [];
        for i = 1:numCom
            adjMat = [adjMat; blockRows{i}];
        end



        %keep the upper triangular part.
        U = triu(adjMat,1);
        A = U + U';
    else
        n2 = n/2;
        P = random('bino', 1, p(1), n2, n2); % upper left block
        P2 = random('bino', 1, p(2), n2, n2); % lower right block
        Q = q*ones(n2,n2);%random('bino', 1, q, n2, n2); % upper right block
        % carve the two triangular and diagonal matrices that we need
        A = [P,Q; Q, P2];
        U = triu(A,1);
        A = U + U';
        A = A - diag(diag(A));
    end

end

