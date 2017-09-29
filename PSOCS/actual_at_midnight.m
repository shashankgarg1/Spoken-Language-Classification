%in this script i took 2 classes +ve and negative
%% -----------------------------------------------------------------------------
clc
clear all
close all

%% ////////////////////////////////////////////////////Data set preparation/////////////////////////////////////////////
 %load iris.txt;
 %x=sortrows(iris,2);
 rng(7)
 x=csvread('3lang_900.csv');
 x(:,79)=(x(:,79)==1);
 %x=iris;
 len=size(x,1);
 %x=sortrows(x,2);
 x=x(randperm(len),:);
 sample=floor(0.7*size(x,1));
 
 for i=1:sample
     train(i,:)=x(i,:);
 end
 
 for i=sample+1:size(x,1)
     test(i-sample,:)=x(i,:);
 end
 
 x=train;
 
 for i=1:(size(x,2)-1)
    H=x(:,i);
    H=H';
    [xf,PS]=mapminmax(H);
    I2(:,i)=xf;
 end
 
 T=x(:,size(x,2));

 T=T';
 [yf,PS5]= mapminmax(T);
 T=yf;
 T=T';

HiddenNodes=15;
Dim=81*HiddenNodes+2;
%Dim=8*HiddenNodes+3;
TrainingNO=size(I2,1);
use=80;
%use=7;

noP=30;           %Number of particles
Max_iteration=500;%Maximum number of iterations
w=2;              %Inirtia weight
wMax=0.9;         %Max inirtia weight
wMin=0.5;         %Min inirtia weight
c1=2;
c2=2;
dt=0.8;
pamin = 0.25;
pamax = 0.75;
alphamin=0.1;
alphamax=1;
k=(1/Max_iteration)*log(alphamin/alphamax);

vel=zeros(noP,Dim); %Velocity vector
pos=zeros(noP,Dim); %Position vector

pBestScore=zeros(noP);
pBest=zeros(noP,Dim);

gBestScore=inf;
gBest=zeros(1,Dim);

ConvergenceCurve=zeros(1,Max_iteration); %Convergence vector

%Initialization
for i=1:size(pos,1) % For each Particle
    for j=1:size(pos,2) % For each dimension
        pos(i,j)=randn();
        vel(i,j)=randn();
        pBest(i,j)=rand(); %%added extra
    end
end

 %initialize gBestScore for min
 gBestScore=inf;

%generate pBestScore from Initialised Pbest
for i=1:size(pos,1)
    for ww=1:(use*HiddenNodes)
        Weights(ww)=pBest(i,ww);
    end
    for bb=use*HiddenNodes+1:Dim
        Biases(bb-(use*HiddenNodes))=pBest(i,bb);
    end        
    fitness=0;
    for pp=1:TrainingNO
        %disp('At training no for pso');
        %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
        %actualvalue=My_FNN(78,HiddenNodes,5,Weights,Biases,I2(pp,:));
        %actualvalue=My_FNN(78,HiddenNodes,3,Weights,Biases,I2(pp,:));
        actualvalue=My_FNN(78,HiddenNodes,2,Weights,Biases,I2(pp,:));
        if(T(pp)==-1)
            fitness=fitness+(1-actualvalue(1))^2;
            fitness=fitness+(0-actualvalue(2))^2;
            
            %fitness=fitness+(0-actualvalue(3))^2;
            %{
            fitness=fitness+(0-actualvalue(4))^2;
            fitness=fitness+(0-actualvalue(5))^2;
            %}
        end
        
        %{
        if(T(pp)==-0.5)
            fitness=fitness+(0-actualvalue(1))^2;
            fitness=fitness+(1-actualvalue(2))^2;
            fitness=fitness+(0-actualvalue(3))^2;
            fitness=fitness+(0-actualvalue(4))^2;
            fitness=fitness+(0-actualvalue(5))^2;
        end
        %}
        
        %{
        if(T(pp)==0)
            fitness=fitness+(0-actualvalue(1))^2;
            
            %{
            fitness=fitness+(0-actualvalue(2))^2;
            fitness=fitness+(1-actualvalue(3))^2;
            fitness=fitness+(0-actualvalue(4))^2;
            fitness=fitness+(0-actualvalue(5))^2;
            %}
            
            fitness=fitness+(1-actualvalue(2))^2;
            fitness=fitness+(0-actualvalue(3))^2;
        end        
        %}
        %{
        if(T(pp)==0.5)
            fitness=fitness+(0-actualvalue(1))^2;
            fitness=fitness+(0-actualvalue(2))^2;
            fitness=fitness+(0-actualvalue(3))^2;
            fitness=fitness+(1-actualvalue(4))^2;
            fitness=fitness+(0-actualvalue(5))^2;
        end        
        
        %}
        
        if(T(pp)==1)
            fitness=fitness+(0-actualvalue(1))^2;
            %fitness=fitness+(0-actualvalue(2))^2;
           
            %{
            fitness=fitness+(0-actualvalue(3))^2;
            fitness=fitness+(0-actualvalue(4))^2;
            fitness=fitness+(1-actualvalue(5))^2;
            %}
            
            fitness=fitness+(1-actualvalue(2))^2;
        end        
        
    end
    fitness=fitness/TrainingNO;
    pBestScore(i)=fitness;
    
    %removed comments since in the algo for cuckoo search gbest is computed
    %added comments again
    %{
    if(gBestScore>fitness)
        gBestScore=fitness;
        gBest=pBest(i,:);
    end
    %}
    
    %{
    if(gBestScore==1)
        break;
    end
    %}
    %added comments end
end

for Iteration=1:Max_iteration
    %disp('At iteration');
    %Calculate MSE
    for i=1:size(pos,1)  
        for ww=1:(use*HiddenNodes)
            Weights(ww)=pos(i,ww);
        end
        for bb=use*HiddenNodes+1:Dim
            Biases(bb-(use*HiddenNodes))=pos(i,bb);
        end        
        fitness=0;
        for pp=1:TrainingNO
            %disp('At training no for cs');
            %disp(['Iteration', num2str(Iteration),' ', num2str(pp),' ', num2str(i)])   
            
            %actualvalue=My_FNN(13,HiddenNodes,5,Weights,Biases,I2(pp,:));
            %actualvalue=My_FNN(78,HiddenNodes,3,Weights,Biases,I2(pp,:));
            actualvalue=My_FNN(78,HiddenNodes,2,Weights,Biases,I2(pp,:));
            %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
            
            if(T(pp)==-1)
                fitness=fitness+(1-actualvalue(1))^2;
                fitness=fitness+(0-actualvalue(2))^2;
                %fitness=fitness+(0-actualvalue(3))^2;
                
                %{
                fitness=fitness+(0-actualvalue(4))^2;
                fitness=fitness+(0-actualvalue(5))^2;
                %}
                
            end
        
            %{
            if(T(pp)==-0.5)
                fitness=fitness+(0-actualvalue(1))^2;
                fitness=fitness+(1-actualvalue(2))^2;
                fitness=fitness+(0-actualvalue(3))^2;
                fitness=fitness+(0-actualvalue(4))^2;
                fitness=fitness+(0-actualvalue(5))^2;
            end
        
            %}
            %{
            if(T(pp)==0)
                fitness=fitness+(0-actualvalue(1))^2;
                
                %{
                fitness=fitness+(0-actualvalue(2))^2;
                fitness=fitness+(1-actualvalue(3))^2;
                fitness=fitness+(0-actualvalue(4))^2;
                fitness=fitness+(0-actualvalue(5))^2;
                %}
                
                fitness=fitness+(1-actualvalue(2))^2;
                fitness=fitness+(0-actualvalue(3))^2;
            end        
            %}
        
            %{
            if(T(pp)==0.5)
                fitness=fitness+(0-actualvalue(1))^2;
                fitness=fitness+(0-actualvalue(2))^2;
                fitness=fitness+(0-actualvalue(3))^2;
                fitness=fitness+(1-actualvalue(4))^2;
                fitness=fitness+(0-actualvalue(5))^2;
            end        
            %}
            
            if(T(pp)==1)
                fitness=fitness+(0-actualvalue(1))^2;
                %fitness=fitness+(0-actualvalue(2))^2;
                
                %{
                fitness=fitness+(0-actualvalue(3))^2;
                fitness=fitness+(0-actualvalue(4))^2;
                fitness=fitness+(1-actualvalue(5))^2;
                %}
                
                fitness=fitness+(1-actualvalue(2))^2;
            end        
        
        end
        fitness=fitness/TrainingNO;
        
        %if Iteration==1
            %pBestScore(i)=fitness;
        %end
        
        %removed comments
        if(pBestScore(i)>fitness)
            pBestScore(i)=fitness;
            pBest(i,:)=pos(i,:);
        end
        
        
        if(gBestScore>fitness)
            gBestScore=fitness;
            gBest=pos(i,:);
        end
        
        %removed comments end
        
        %{
        this should be zero -- check
        if(gBestScore==1)
            break;
        end
        %}
        
    end

    %Update the w of PSO
    w=wMin-Iteration*(wMax-wMin)/Max_iteration;
    
    %Update the velocity and position of particles
    for i=1:size(pos,1)
        for j=1:size(pos,2)       
            vel(i,j)=w*vel(i,j)+c1*rand()*(pBest(i,j)-pos(i,j))+c2*rand()*(gBest(j)-pos(i,j));
            pos(i,j)=pos(i,j)+dt*vel(i,j);
        end
    end
    
    
    pa=pamax-((Iteration/Max_iteration)*(pamax-pamin));
    alpha=alphamax*exp(k*Iteration);
    new_sol=floor(pa*noP);
    
    pBestScoreArray(1:noP,1)=pBestScore(1:noP);
    pBestScoreArray(1:noP,2)=1:noP;
    
    pBestScoreArray=sortrows(pBestScoreArray,-1);
    pBestScoreArray=pBestScoreArray(1:new_sol,:);
    
    for i=1:new_sol
        %disp('At new sol');
        index=pBestScoreArray(i,2);
        pBest(index,:)=levy_flight(pBest(index,:),gBest,alpha); %%changed gBestScore to gBest as according to algo. Testing accuracy dropped to 88% after doing this
        pBestScore(index)=compPBestScore(pBest(index,:), HiddenNodes, Dim, TrainingNO, I2, T);
        %{
        if(gBestScore>pBestScore(index))
            gBestScore=pBestScore(index);
            gBest=pBest(index,:);
        end
        %}
    end
    
    for i=1:noP
        if(gBestScore>pBestScore(i))
            gBestScore=pBestScore(i);
            gBest=pBest(i,:);
        end
    end
  
    
    %{
    This doesn't work-- both accuracy down to 0%
    new_pBest=empty_nests(pBest,pa) ;
    [fnew,best,pBest,pBestScore]=get_best_nest(pBest,new_pBest,pBestScore,HiddenNodes, Dim, TrainingNO, I2, T); %best is curr pbest and fnew is curr pbestscore
    if(gBestScore>fnew(1))
        gBestScore=fnew(1);
        gBest=best;
    end
    %}
    ConvergenceCurve(1,Iteration)=gBestScore;
    
    disp(['PSOCS is training FNN (Iteration = ', num2str(Iteration),' ,MSE = ', num2str(gBestScore),')'])        
end

%% ///////////////////////Calculate the classification//////////////////////
        Rrate=0;
        Weights=gBest(1:use*HiddenNodes);
        Biases=gBest(use*HiddenNodes+1:Dim);
         for pp=1:TrainingNO
            %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
            %actualvalue=My_FNN(13,HiddenNodes,5,Weights,Biases,I2(pp,:));
            %actualvalue=My_FNN(78,HiddenNodes,3,Weights,Biases,I2(pp,:));
            actualvalue=My_FNN(78,HiddenNodes,2,Weights,Biases,I2(pp,:));
            if(T(pp)==-1)
                %if (round(actualvalue(1))==1 && round(actualvalue(2))==0 && round(actualvalue(3))==0 && round(actualvalue(4))==0 && round(actualvalue(5))==0)
                if (round(actualvalue(1))==1 && round(actualvalue(2))==0)
                %if (round(actualvalue(1))==1 && round(actualvalue(2))==0 && round(actualvalue(3))==0)
                    Rrate=Rrate+1;
                end
            end
            
            %{
            if(T(pp)==-0.5)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==1 && round(actualvalue(3))==0 && round(actualvalue(4))==0 && round(actualvalue(5))==0)
                    Rrate=Rrate+1;
                end  
            end
            %}
            
            %{
            if(T(pp)==0)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==1 && round(actualvalue(3))==0)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==1 && round(actualvalue(4))==0 && round(actualvalue(5))==0)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==1)
                    Rrate=Rrate+1;
                end              
            end
            %}
            
            %{
            if(T(pp)==0.5)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==1)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==0 && round(actualvalue(4))==1 && round(actualvalue(5))==0)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==1)
                    Rrate=Rrate+1;
                end              
            end
            %}
            
            if(T(pp)==1)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==1)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==0 && round(actualvalue(4))==0 && round(actualvalue(5))==1)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==1)
                    Rrate=Rrate+1;
                end              
            end
            
        end
        
ClassificationRate=(Rrate/TrainingNO)*100;
disp(['Training Classification rate = ', num2str(ClassificationRate)]);

%% Draw the convergence curve
hold on;      
semilogy(ConvergenceCurve);
title(['Training Classification rate : ', num2str(ClassificationRate), '%']); 
xlabel('Iteration');
ylabel('MSE');
box on
grid on
axis tight
hold off;

%%


%code added
clear I2;
 x=test;
 TrainingNO=size(x,1);
 for i=1:(size(x,2)-1)
    H=x(:,i);
    H=H';
    [xf,PS]=mapminmax(H);
    I2(:,i)=xf;
 end
 
 T=x(:,size(x,2));
 
 Thelp=T;
 T=T';
 [yf,PS5]= mapminmax(T);
 T=yf;
 T=T';
 
 
 %% ///////////////////////Calculate the classification//////////////////////
        Rrate=0;
        Weights=gBest(1:use*HiddenNodes);
        Biases=gBest(use*HiddenNodes+1:Dim);
         for pp=1:TrainingNO
            %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
            %actualvalue=My_FNN(13,HiddenNodes,5,Weights,Biases,I2(pp,:));
            %actualvalue=My_FNN(78,HiddenNodes,3,Weights,Biases,I2(pp,:));
            actualvalue=My_FNN(78,HiddenNodes,2,Weights,Biases,I2(pp,:));
            
            if(T(pp)==-1)
                %if (round(actualvalue(1))==1 && round(actualvalue(2))==0 && round(actualvalue(3))==0 && round(actualvalue(4))==0 && round(actualvalue(5))==0)
                if (round(actualvalue(1))==1 && round(actualvalue(2))==0)
                %if (round(actualvalue(1))==1 && round(actualvalue(2))==0 && round(actualvalue(3))==0)
                    Rrate=Rrate+1;
                end
            end
            %{
            if(T(pp)==-0.5)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==1 && round(actualvalue(3))==0 && round(actualvalue(4))==0 && round(actualvalue(5))==0)
                    Rrate=Rrate+1;
                end  
            end
            %}
            
            %{
            if(T(pp)==0)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==1 && round(actualvalue(3))==0)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==1 && round(actualvalue(4))==0 && round(actualvalue(5))==0)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==1)
                    Rrate=Rrate+1;
                end              
            end
            %}
            
            %{
            if(T(pp)==0.5)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==1)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==0 && round(actualvalue(4))==1 && round(actualvalue(5))==0)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==1)
                    Rrate=Rrate+1;
                end              
            end
            %}
            
            if(T(pp)==1)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==1)
                %if (round(actualvalue(1))==0 && round(actualvalue(2))==0 && round(actualvalue(3))==0 && round(actualvalue(4))==0 && round(actualvalue(5))==1)
                if (round(actualvalue(1))==0 && round(actualvalue(2))==1)
                    Rrate=Rrate+1;
                end              
            end
            
            
        end
        
ClassificationRate=(Rrate/TrainingNO)*100;
disp(['Testing Classification rate = ', num2str(ClassificationRate)]);

%% Draw the convergence curve
hold on;      
semilogy(ConvergenceCurve);
title(['Testing Classification rate : ', num2str(ClassificationRate), '%']); 
xlabel('Iteration');
ylabel('MSE');
box on
grid on
axis tight
hold off;



%code ends
