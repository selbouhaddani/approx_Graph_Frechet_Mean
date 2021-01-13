%This code is designed to run for equal sized communities but can be
%generalized to arbitrary sized communities. In the context of the paper
%this code is associated with, the parameter s of the stochastic block models is constant.

%load the data set
load("...\Small_World_Data.mat"); %Change this to the other sets of data provided or your own set.


sampleEigSet = cell(N,1); %We only need to store the sample eigenvalues since this is the only data needed for the computations
avgEigVal = zeros(n,1);
avgDeg = 0;

for i = 1:N
    sampleEigSet{i} = eig(sampleAdjSet{i});
    avgEigVal = avgEigVal + sampleEigSet{i};
end

avgEigVal = avgEigVal/N;

c_star = detectCom(avgEigVal,n);


while n/c_star ~= floor(n/c_star) 
    c_star = c_star + 1;
end

avgEigVal = zeros(c_star,1);

for i = 1:N
    A = sampleAdjSet{i};
    E = eig(A);                                 %compute largest c eigen values
    
    E = sort(E(end-c_star+1:end));
    
    sampleEigSet{i} = abs(E);                   %sort eigenvalues
    avgEigVal = avgEigVal+sampleEigSet{i};      %determine the average eigenvalue of the set
    
    deg = sum(A);
    dbar = sum(deg)/n;
    avgDeg = avgDeg + dbar;                     %determine average density (rho_n)
end

avgDeg = avgDeg/N;                              %average density
avgEigVal = sort(sum(avgEigVal,2)/N,'descend'); %average eigen value. This vector minimizes the objective for the unconstrained and truncated problem


%Gradient Descent

%Initialize p and q
comSize = n/c_star;
p_i = (0.2:(0.6-0.2)/c_star:0.6)';              
p_i = p_i(1:c_star);  %initial p (important to initalize each p_i(j) such that for all j,k we have p_i(j) ~= p_i(k)
q_i = (n*avgDeg - comSize*(comSize-1)*sum(p_i))/(comSize*c_star*(n-comSize)); %initial q


%If q_i > p_i(j) for some j, increase p_i(j) until p_i(j) > q_i for all j
while(min(p_i) < q_i)
    [mini,min_index] = min(p_i);
    p_i(min_index) = mini + (1-mini)/2;
    q_i = (n*avgDeg - comSize*(comSize-1)*sum(p_i))/(comSize*c_star*(n-comSize)); %update q_i
end



%If q_i < 0 then the values in p_i are too large. Decrement these until
%q_i > 0
while(q_i < 0)
    [maxi,max_index] = max(p_i);
    p_i(max_index) = maxi - maxi/2;
    q_i = (n*avgDeg - comSize*(comSize-1)*sum(p_i))/(comSize*c_star*(n-comSize)); %update q_i
end

if min(p_i) < 0 || max(p_i) > 1 || q_i < 0 || q_i > 1
    disp("No good Frechet mean exists. Try different c_star or different initial guess at p_i.")
    return
end

%Now p_i is initialized such that it is within the feasible set of
%solutions. (Exists on the line of constant sparsity and within the bounds
%of sparsity)


%Implementation of gradient descent
%Stopping condition: relative change in p_i
p_prev = 2*p_i;
maxSteps = 100;
curStep = 1;
h = 0.005; %Having a variable stepsize would be beneficial. Large steps early and smaller steps as the algorithm converges

%Store information for plotting and analysis
ps = p_i;
qs = q_i;

objVals = [];
grads = [];

%Begin gradient descent

while(norm(p_i-p_prev)/norm(p_prev) > 1*10^-4 && curStep < maxSteps) %Various different stopping condition can be used
    p_prev = p_i;
    if curStep == 1 
        %if this is the first step, seed the newton method for approximating the eigenvalues with a bad initial guess
        %and iterate for longer
        guess = -100*ones(c_star,1); 
    end
    
    %compute the approximate eigenvalues of stochastic block model
    eigs_i = eigsApprox(p_i,q_i,n,guess);
    
    %seed future guess of the approx eigenvaleus with the previously found
    %eigenvalues. (small change in parameter leads to small change in
    %eigenvalues so the previous guess will be close to the root
    guess = eigs_i;
    
    %compute the objective value
    obj_p_i = obj(avgEigVal,eigs_i);
    objVals = [objVals;obj_p_i];
    
    %Compute the gradient direction
    %Since this code is intended as a proof of concept we implement a basic
    %numerical approximation of the gradient
    grad_p_i = gradObj(avgEigVal,N,p_i,h,n,avgDeg,guess);
    grads = [grads,grad_p_i];
    
    %Compute linesearch in direction of gradient
    [gamma,eigs_i] = linesearch(p_i,grad_p_i,obj_p_i,avgEigVal,N,n,avgDeg,guess);
    guess = eigs_i; %this will seed the guess for future Newton's method approximations of eigenvalues
    
    %Update p_i
    p_i = p_i - gamma*grad_p_i;
    
    %Check that a valid step was taken. If not, project back onto the
    %feasible set. (proximal gradient descent)
    
    %Since feasible set is a square this projection is easy
    if max(p_i) > 1
        [~,indexMax] = max(p_i);
        p_i(indexMax) = 1;
    end
    q_i = (n*avgDeg - comSize*(comSize-1)*sum(p_i))/(comSize*c_star*(n-comSize));

    if min(p_i) < 0
        [~,indexMin] = min(p_i);
        p_i(indexMin) = q_i; %project to previous q_i then when computing next q_i this will be a feasible point
        q_i = (n*avgDeg - comSize*(comSize-1)*sum(p_i))/(comSize*c_star*(n-comSize));
    end
    
    if q_i > min(p_i)
        disp("Frechet mean is too dense")
        return 
    end
    ps = [ps,p_i];
    qs = [qs,q_i];
    curStep = curStep + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [objective] = obj(avgEigVal,eigs)
    objective = norm(eigs - avgEigVal(1:length(eigs)),2);
end


function [c] = detectCom(avgEig,n)
    i = n;
    
    inBulk = isInBulk(avgEig,i);
    
    while inBulk == false
        i = i-1;
        inBulk = isInBulk(avgEig,i);
    end
    
    c = n-i;
end

function [inBulk] = isInBulk(avgEigs, n)
    inBulk = true;
    r = avgEigs(n); %set radius for semicircle distribution
    
    s = @(x) 2/(pi*r^2)*(r^2-x.^2).^0.5; %pdf of semicircle
    S = @(x) 1/pi*(asin(x./r)+x./r.*(r^2-x.^2).^0.5./r+pi/2); %cdf of semicircle
    
    %Define the moments for the various order statistics. Here we use the
    %three largest order statistics X_(n), X_(n-1), X_(n-2) which are the
    %4th, 3rd, and 2nd largest eigenvalues. The largest eigenvalue defines
    %the edge of the bulk
    
    mo11 = @(x) x.*(n-1).*s(x).*S(x).^(n-2);
    mo12 = @(x) x.^2.*(n-1).*s(x).*S(x).^(n-2);
    
    mo21 = @(x) x.*(n-1)*(n-2).*s(x).*S(x).^(n-3).*(1-S(x));
    mo22 = @(x) x.^2.*(n-1)*(n-2).*s(x).*S(x).^(n-3).*(1-S(x));
    
    mo31 = @(x) x.*((n-1)*(n-2)*(n-3)/2.*s(x).*S(x).^(n-4).*(1-S(x)).^(2));
    mo32 = @(x) x.^2.*((n-1)*(n-2)*(n-3)/2.*s(x).*S(x).^(n-4).*(1-S(x)).^(2));
    
    %Numerically determine the moments
    mo11_val = integral(@(x) mo11(x),-r,r);
    mo12_val = integral(@(x) mo12(x),-r,r);
    Var1 = mo12_val - mo11_val^2;
    
    mo21_val = integral(mo21,-r,r);
    mo22_val = integral(mo22,-r,r);
    Var2 = mo22_val - mo21_val^2;
    
    
    mo31_val = integral(mo31,-r,r);
    mo32_val = integral(mo32,-r,r);
    Var3 = mo32_val - mo31_val^2;
    
    %This can be generalized to compute an arbitrary number of order
    %statistics but matlab does not simplify them well when written in a
    %for loop. A fix is to hard code in the finite number of order
    %statistics as is done above.
    if abs(mo11_val - avgEigs(n-1)) > sqrt(Var1) || abs(mo21_val - avgEigs(n-2)) > sqrt(Var2) || abs(mo31_val - avgEigs(n-3)) > sqrt(Var3)
        inBulk = false;
    end
    
end

function [grad] = gradObj(avgEigVal,N,p_i,h,n,avgDeg,guess)

    [c,~] = size(p_i);
    comSize = n/c;
    grad = zeros(c,1);
    
    for i = 1:c
        e_i = zeros(c,1);
        e_i(i) = h;
        p_i_p = p_i + e_i;
        
        p_i_m = p_i - e_i;
        if p_i_p(i) > 1
            p_i_p(i) = 1;
        end
        q_i_p = (n*avgDeg - comSize*(comSize-1)*sum(p_i_p))/(comSize*c*(n-comSize));

        if p_i_m < 0
            p_i_m(i) = 0;
        end
        q_i_m = (n*avgDeg - comSize*(comSize-1)*sum(p_i_m))/(comSize*c*(n-comSize));

        if q_i_m > min(p_i_m)
            disp("Meaningful gradient couldn't be computed")
            return
        end
        
        eigs_pp = eigsApprox(p_i_p,q_i_p,n,guess);
        eigs_pm = eigsApprox(p_i_m,q_i_m,n,guess);
        
        obj_pp = obj(avgEigVal,eigs_pp);
        obj_pm = obj(avgEigVal,eigs_pm);
        
        grad(i) = (obj_pp - obj_pm)/(2*h);

    end    
    
end

function [gamma,eigs] = linesearch(p,grad_p,obj_p,avgEigVal,N,n,avgDeg,guess)
    bound = abs(1/max(grad_p));
    curMin = obj_p;
    gamma = 0;
    [c,~] = size(p);
    comSize = n/c;
        
    for g = 0:bound/2000:bound
        pg = p - g*grad_p;
        if max(pg) > 1
            [~,index] = max(pg);
            pg(index) = 1;
        end
        if min(pg) < 0
            [~,indexMin] = min(pg);
            pg(indexMin) = 0;
        end
        
        q = (n*avgDeg - comSize*(comSize-1)*sum(pg))/(comSize*c*(n-comSize));
        if q < 0
            disp("Line of constant expected density graphs leaves the searchable set")
            return
        end
        
        if q > min(pg)
            disp("Can't move through dense graphs to get to frechet mean")
            return
        end
        eigs_pg = eigsApprox(pg,q,n,guess);
        obj_g = obj(avgEigVal,eigs_pg);
        if obj_g <= curMin || g == 0
            curMin = obj_g;
            gamma = g;
            eigs = eigs_pg;
        end
        if obj_g > curMin
            break
        end
        
    end
        
end

function [t] = R_prime(EV,z,Moments,d,k)
    t = 0;
    
    for l = 0:d
        if l == 0
            t = t+-z^(-2)*EV(:,k)'*eye(size(Moments{1}))*EV(:,k);
        end
        if l > 1
            t = t+(-(l+1))*z^(-(l+1)-1)*EV(:,k)'*Moments{l}*EV(:,k);
        end 
    end
    t = -t;
end

function [t] = R(EV,z,Moments,d,k)
    t = 0;
    for l = 0:d
        if l == 0
            t = t+z^(-1)*EV(:,k)'*eye(size(Moments{1}))*EV(:,k);
        end
        if l > 1
            t = t+z^(-(l+1))*EV(:,k)'*Moments{l}*EV(:,k);
        end 
    end
    t = -t;
end

function [eigs_i] = eigsApprox(p_i,q_i,n,guess)
    [c,~] = size(p_i);

    %compute the expected adj matrix
    EA = exp_rand_adj(p_i,q_i,n);
    [EV,ED] = eigs(EA,c);

    Moments = cell(4,1);
    Moments{1} = EA;
    Moments{2} = var_rand_adj(p_i,q_i,n);
    Moments{3} = third_rand_adj(p_i,q_i,n);
    Moments{4} = fourth_rand_adj(p_i,q_i,n);
    
    eigs_i = zeros(c,1);
    for k = 1:c
        %%%Newton method
        iter2 = 750;
        d_k = ED(k,k);
        
        %Bounds for initial guess
        c_0 = 0.5;                              %This hyperparam must be less than 1
        if d_k > 0
            a_k = d_k/(1+c_0/2);
            b_k = (1+c_0/2)*d_k;
        else
            a_k = (1+c_0/2)*d_k;
            b_k = d_k/(1+c_0/2);
        end
        if guess == -100*ones(c,1)
            t0 = (a_k + b_k)/2;
            iter2 = 750;
        else
            t0 = guess(k);
            iter2 = 50;
        end
        for j = 1:iter2
            t_next = t0 - (1+d_k*R(EV,t0,Moments,4,k))/(d_k*R_prime(EV,t0,Moments,4,k));
            t0 = t_next;
        end
        eigs_i(k) = t_next;
    end
    
end


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

function [A] = exp_rand_adj(p,q,n)
    %c is the num of communities
    numCom = length(p);
    qvec = q*ones(numCom*(numCom-1)/2,1);
    comSize = n/numCom;
    if numCom > 2
        comGraphs = cell(numCom,1);

        for i = 1:numCom
            comGraphs{i} = p(i)*ones(comSize,comSize);%random('bino', 1, p(i), comSize, comSize);
        end

        numInterCom = length(qvec);
        interComGraphs = cell(numInterCom,1);

        for i = 1:numInterCom
            interComGraphs{i} = qvec(i)*ones(comSize,comSize);%random('bino', 1, qvec(i), comSize, comSize);
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
        P = p(1)*ones(n2,n2);%random('bino', 1, p(1), n2, n2); % upper left block
        P2 = p(2)*ones(n2,n2);%random('bino', 1, p(2), n2, n2); % lower right block
        Q = q*ones(n2,n2);%random('bino', 1, q, n2, n2); % upper right block
        % carve the two triangular and diagonal matrices that we need
        A = [P,Q; Q, P2];
        U = triu(A,1);
        A = U + U';
        A = A - diag(diag(A));
    end

end

function [A] = var_rand_adj(p,q,n)
    %c is the num of communities
    numCom = length(p);
    if numCom > 2
        qvec = q*ones(numCom*(numCom-1)/2,1);
        comSize = n/numCom;

        comGraphs = cell(numCom,1);

        for i = 1:numCom
            comGraphs{i} = p(i)*(1-p(i))*ones(comSize,comSize);%random('bino', 1, p(i), comSize, comSize);
        end

        numInterCom = length(qvec);
        interComGraphs = cell(numInterCom,1);

        for i = 1:numInterCom
            interComGraphs{i} = qvec(i)*(1-qvec(i))*ones(comSize,comSize);%random('bino', 1, qvec(i), comSize, comSize);
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
        P = p(1)*(1-p(1))*ones(n2,n2);%random('bino', 1, p(1), n2, n2); % upper left block
        P2 = p(2)*(1-p(2))*ones(n2,n2);%random('bino', 1, p(2), n2, n2); % lower right block
        Q = q*(1-q)*ones(n2,n2);%random('bino', 1, q, n2, n2); % upper right block
        % carve the two triangular and diagonal matrices that we need
        A = [P,Q; Q, P2];
        U = triu(A,1);
        A = U + U';
        A = A - diag(diag(A));
    end
end

function [A] = third_rand_adj(p,q,n)
    %c is the num of communities
    numCom = length(p);
    if numCom > 2
        qvec = q*ones(numCom*(numCom-1)/2,1);
        comSize = n/numCom;

        comGraphs = cell(numCom,1);

        for i = 1:numCom
            comGraphs{i} = (2*p(i)^3-3*p(i)^2+p(i))*ones(comSize,comSize);%random('bino', 1, p(i), comSize, comSize);
        end

        numInterCom = length(qvec);
        interComGraphs = cell(numInterCom,1);

        for i = 1:numInterCom
            interComGraphs{i} = (2*qvec(i)^3-3*qvec(i)^2+qvec(i))*ones(comSize,comSize);%random('bino', 1, qvec(i), comSize, comSize);
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
        P = (2*p(1)^3-3*p(1)^2+p(1))*ones(n2,n2);%random('bino', 1, p(1), n2, n2); % upper left block
        P2 = (2*p(2)^3-3*p(2)^2+p(2))*ones(n2,n2);%random('bino', 1, p(2), n2, n2); % lower right block
        Q = (2*q^3-3*q^2+q)*ones(n2,n2);%random('bino', 1, q, n2, n2); % upper right block
        % carve the two triangular and diagonal matrices that we need
        A = [P,Q; Q, P2];
        U = triu(A,1);
        A = U + U';
        A = A - diag(diag(A));
    end

end

function [A] = fourth_rand_adj(p,q,n)
    %c is the num of communities
    numCom = length(p);
    if numCom > 2
        qvec = q*ones(numCom*(numCom-1)/2,1);
        comSize = n/numCom;

        comGraphs = cell(numCom,1);

        for i = 1:numCom
            comGraphs{i} = (-3*p(i)^4+6*p(i)^3-4*p(i)^2+p(i))*ones(comSize,comSize);%random('bino', 1, p(i), comSize, comSize);
        end

        numInterCom = length(qvec);
        interComGraphs = cell(numInterCom,1);

        for i = 1:numInterCom
            interComGraphs{i} = (-3*qvec(i)^4+6*qvec(i)^3-4*qvec(i)^2+qvec(i))*ones(comSize,comSize);%random('bino', 1, qvec(i), comSize, comSize);
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
        P = (-3*p(1)^4+6*p(1)^3-4*p(1)^2+p(1))*ones(n2,n2);%random('bino', 1, p(1), n2, n2); % upper left block
        P2 = (-3*p(2)^4+6*p(2)^3-4*p(2)^2+p(2))*ones(n2,n2);%random('bino', 1, p(2), n2, n2); % lower right block
        Q = (-3*q^4+6*q^3-4*q^2+q)*ones(n2,n2);%random('bino', 1, q, n2, n2); % upper right block
        % carve the two triangular and diagonal matrices that we need
        A = [P,Q; Q, P2];
        U = triu(A,1);
        A = U + U';
        A = A - diag(diag(A));
    end
    
end
