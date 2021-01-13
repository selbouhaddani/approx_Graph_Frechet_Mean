data = zeros(3,6);
for i = 1:6
    data(1,i) = plottingData{i,2}(1);
    data(2,i) = plottingData{i,2}(2);
    data(3,i) = plottingData{i,2}(3);
end

p = [p1_samples;p2_samples;p3_samples];
t = [sort(t_samples);sort(t_samples);sort(t_samples)];


figure(1)
plot(t,p,'.','color','black')
hold on
plot([0:0.2:1,0:0.2:1,0:0.2:1],[data(1,:),data(2,:),data(3,:)],'x','color','black')

legend('Sampled Values','Estimated Values')
distVec = zeros(N-1,1);
for i = 1:N-1
    distVec(i) = norm(sampleEigSet{i+1}-sampleEigSet{i})/(t_samples(i+1)-t_samples(i));
end








