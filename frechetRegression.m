%Simple Regression Experiment
%clear all

load("FILEPATH\regression_data.mat");


c_star = c_sample; %Use oracle knowledge. Assume constant value of c_star across sample.
sampleEigSet = cell(N,1); %We only need to store the sample eigenvalues since this is the only data needed for the computations

for i = 1:N                           
    sampleEigSet{i} = abs(sort(eigs(sampleAdjSet{i},c_star)));
end

t_bar = mean(t_samples);
t_var = var(t_samples);

stepSize = 0.2;

plottingData = cell(numel(0:stepSize:1),4);
index = 1;
for time = 0:stepSize:1
    %For each t, compute the weighted sample frechet mean
    %Step 1: compute the weighted average density and weighted average
    %eigenvalues
    weightedRho = weightedDensity(t_samples,t_bar,t_var,sampleAdjSet,time,n);
    weightedAvgEig = weightedEig(t_samples,t_bar,t_var,sampleEigSet,time);
    
    avgDeg = n*weightedRho;
    avgEigVal = sort(weightedAvgEig,'descend');
    
    %Perform gradient descent
    %Gradient Descent

    %initialize p and q
    comSize = n/c_star;
    p_i = (0.2:(0.5-0.2)/c_star:0.5)';              
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
    %of constant sparsity)


    %implementation of gradient descent
    %stopping condition: relative change in p_i
    p_prev = 2*p_i;
    maxSteps = 100;
    curStep = 1;
    h = 0.005; %Having a variable stepsize would be beneficial. Large steps early and smaller steps as the algorithm converges

    %Store information for plotting and analysis

    ps = p_i;
    qs = q_i;

    objVals = [];
    grads = [];

    %begin gradient descent for each time

    while(norm(p_i-p_prev)/norm(p_prev) > 1*10^-4 && curStep < maxSteps) %Various different stopping condition can be used
        p_prev = p_i;
        if curStep == 1 
            %if this is the first step, seed the newton method for approximating the eigenvalues with a bad initial guess
            %and iterate for longer
            guess = -100*ones(c_star,1); 
        end

        %compute the approximate eigenvalues of SBM(p_i,q_i)
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

        %check that a valid step was taken. If not, project back onto the
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
            q_i = (n*avgDeg - comSize*(comSize-1)*sum(p_i))/(comSize*c_star*(n-comSize)); %(Do I have to worry about this?)
        end

        if q_i > min(p_i)
            disp("Frechet mean is too dense")
            return 
        end
        ps = [ps,p_i];
        qs = [qs,q_i];
        curStep = curStep + 1;
        norm(p_i-p_prev)/norm(p_prev)
    end
    
    %save time, p_i, and q_i in a cell array
    plottingData{index,1} = time;
    plottingData{index,2} = p_i;
    plottingData{index,3} = q_i;
    plottingData{index,4} = objVals;
    index = index + 1;
    
 end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [avgEig] = weightedEig(t_samples,t_bar,t_var,sampleEigSet,t)
    c_star = 3;
    avgEig = zeros(c_star,1);
    N = numel(sampleEigSet);
    factor = 0;
    for i = 1:N
        avgEig = avgEig + s_weight(t_samples, t_bar, t_var, i, t)*sampleEigSet{i};
        factor = factor + s_weight(t_samples, t_bar, t_var, i, t);
    end
    avgEig = avgEig/factor;
end

function [rho] = weightedDensity(t_samples,t_bar,t_var,sampleAdjSet,t,n)
    rho = 0;
    factor = 0;
    N = numel(sampleAdjSet);
    for i = 1:N
        rho = rho + s_weight(t_samples, t_bar, t_var, i, t)*sum(sum(sampleAdjSet{i}))/(n*(n-1));
        factor  = factor + s_weight(t_samples, t_bar, t_var, i, t);
    end
    rho = rho/factor;
    
end

function [weight] = s_weight(t_samples, t_bar, t_var, i,t)
    weight = 1+(t_samples(i) - t_bar)*t_var^(-1)*(t-t_bar);
end


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
    
    c = n-i; %alg terminates when observing 2 eigenvalues from bulk.
end

function [inBulk] = isInBulk(avgEigs, n)
    inBulk = true;
    r = avgEigs(n); %set radius for semicircle distribution
    
    s = @(x) 2/(pi*r^2)*(r^2-x.^2).^0.5; %pdf of semicircle
    S = @(x) 1/pi*(asin(x./r)+x./r.*(r^2-x.^2).^0.5./r+pi/2); %cdf of semicircle
    
    %Consider using a different density to approximate the bulk. Such as  a
    %triangle as below. 
    
    %s = @(x) -abs(x)/(r^2) + 1/r;
    %S = @(x) -1/(2*r^2)*x.^2.*sign(x)+x/(r)+1/2;
    
    
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
        curMin
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
    %A = rand_adj(p_i,q_i,n);
    [c,~] = size(p_i);
  
    %[V,D] = eigs(A,c);
    %D = diag(D);
    %compute the expected adj matrix
    EA = exp_rand_adj(p_i,q_i,n);
    
    [EV,ED] = eigs(EA,c);
    %W = A-EA;
    %[WV,WD] = eigs(W,c);
    %WD = diag(WD);

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
        %bounds for initial guess
        c_0 = 0.5; %hyperparam < 1
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

function [A] = var_rand_adj(p,q,n)
    [numCom,~] = size(p);
    
    largestCom = floor(n/numCom) + rem(n,numCom);
    otherCom = floor(n/numCom);
    
    comSizes = otherCom*ones(1,numCom);
    comSizes(1) = largestCom;
    
    c = cell(numel(comSizes),1);
    for i = 1:numel(comSizes)
        c{i} = (p(i)*(1-p(i))-q*(1-q))*ones(comSizes(i),comSizes(i));
    end
    
    A = blkdiag(c{:})+q*(1-q);
    A = A - diag(diag(A));

end

function [A] = third_rand_adj(p,q,n)
    [numCom,~] = size(p);
    
    largestCom = floor(n/numCom) + rem(n,numCom);
    otherCom = floor(n/numCom);
    
    comSizes = otherCom*ones(1,numCom);
    comSizes(1) = largestCom;
    
    c = cell(numel(comSizes),1);
    for i = 1:numel(comSizes)
        c{i} = (2*p(i)^3-3*p(i)^2+p(i)-(2*q^3-3*q^2+q))*ones(comSizes(i),comSizes(i));
    end
    
    A = blkdiag(c{:})+(2*q^3-3*q^2+q);
    A = A - diag(diag(A));

end

function [A] = fourth_rand_adj(p,q,n)
    [numCom,~] = size(p);
    
    largestCom = floor(n/numCom) + rem(n,numCom);
    otherCom = floor(n/numCom);
    
    comSizes = otherCom*ones(1,numCom);
    comSizes(1) = largestCom;
    
    c = cell(numel(comSizes),1);
    for i = 1:numel(comSizes)
        c{i} = ((-3*p(i)^4+6*p(i)^3-4*p(i)^2+p(i))-(-3*q^4+6*q^3-4*q^2+q))*ones(comSizes(i),comSizes(i));
    end
    
    A = blkdiag(c{:})+(-3*q^4+6*q^3-4*q^2+q);
    A = A - diag(diag(A));

end






