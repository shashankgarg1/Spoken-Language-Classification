  clc
  clear all
  close all

% Data set preparation

  [x,y]=read_data('glass.data');
  output_classes=unique(y);
  
% splitting into training and testing
  sample=floor(0.7*size(x,1));
  train_x=zeros(sample,size(x,2));
  train_y=zeros(1,sample);  
  for i=1:sample
    train_x(i,:)=x(i,:);
    train_y(i)=y(i);
  end

  test_x=zeros(size(x,1)-sample,size(x,2));
  test_y=zeros(1,size(x,1)-sample);
  for i=sample+1:size(x,1)
    test_x(i-sample,:)=x(i,:);
    test_y(i-sample)=y(i);
  end

% from now on use train_x instead of I2

  HiddenNodes=10;
  len=length(unique(train_y));
  columns=size(train_x,2);
  Dim=(columns+1+len)*HiddenNodes+len;

% Dim=8*HiddenNodes+3;

  TrainingNO=size(train_x,1);
  use=columns+len;

% use=7;

  noP=30;           %Number of particles
  Max_iteration=25000;%Maximum number of iterations
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
  k=(1.0/Max_iteration)*log(alphamin/alphamax);

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
      pos(i,j)=rand();
      vel(i,j)=rand();
      pBest(i,j)=rand(); %%added extra
    end
  end

  %initialize gBestScore for min
  gBestScore=inf;

  Weights=zeros(1,use*HiddenNodes);
  Biases=zeros(1,Dim-use*HiddenNodes);
  
  weight_limit=use*HiddenNodes;
  
  %generate pBestScore from Initialised Pbest
  for i=1:size(pos,1)
    for ww=1:(weight_limit)
      Weights(ww)=pBest(i,ww);
    end
    
    for bb=weight_limit+1:Dim
      Biases(bb-(weight_limit))=pBest(i,bb);
    end
    
    fitness=0;
    for pp=1:TrainingNO
      %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
      actualvalue=My_FNN(columns,HiddenNodes,len,Weights,Biases,train_x(pp,:));
      out=(output_classes==train_y(pp));
      fitness=fitness+sum((actualvalue-out').^2);
    end
    fitness=fitness/TrainingNO;
    pBestScore(i)=fitness;
    
    if(gBestScore>fitness)
      gBestScore=fitness;
      gBest=pBest(i,:);
    end
    
  end

  for Iteration=1:Max_iteration
    %Calculate MSE
    for i=1:size(pos,1)
      for ww=1:(weight_limit)
        Weights(ww)=pos(i,ww);
      end
      
      for bb=weight_limit+1:Dim
        Biases(bb-(weight_limit))=pos(i,bb);
      end
      
      fitness=0;
      for pp=1:TrainingNO
        %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
        actualvalue=My_FNN(columns,HiddenNodes,len,Weights,Biases,train_x(pp,:));
        out=(output_classes==train_y(pp));
        fitness=fitness+sum((actualvalue-out').^2);
      end
      fitness=fitness/TrainingNO;
      
      if(pBestScore(i)>fitness)
        pBestScore(i)=fitness;
        pBest(i,:)=pos(i,:);
      end
        
      if(gBestScore>fitness)
        gBestScore=fitness;
        gBest=pos(i,:);
      end
        
      if(gBestScore==0)
        break;
      end
      
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
      index=pBestScoreArray(i,2);
      pBest(index,:)=levy_flight(pBest(index,:),gBest,alpha);
      pBestScore(index)=compPBestScore(pBest(index,:), HiddenNodes, Dim, TrainingNO,train_x,train_y,len,output_classes);
      if(gBestScore>pBestScore(index))
        gBestScore=pBestScore(index);
        gBest=pBest(index,:);
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

% Calculate the classification for training
  Rrate=0;
  Weights=gBest(1:weight_limit);
  Biases=gBest(weight_limit+1:Dim);
  for pp=1:TrainingNO
%   actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
    actualvalue=My_FNN(columns,HiddenNodes,len,Weights,Biases,train_x(pp,:));
    out=(output_classes==train_y(pp));
    if(sum((actualvalue-out').^2)==0)
      Rrate=Rrate+1;
    end
  end
        
  ClassificationRate=(Rrate/TrainingNO)*100;
  disp(['Training Classification rate = ', num2str(ClassificationRate)]);

% Draw the convergence curve
  hold on;      
  semilogy(ConvergenceCurve);
  title(['Training Classification rate : ', num2str(ClassificationRate), '%']); 
  xlabel('Iteration');
  ylabel('MSE');
  box on
  grid on
  axis tight
  hold off;
  
% Calculate the classification for testing
  Rrate=0;
  Weights=gBest(1:weight_limit);
  Biases=gBest(weight_limit+1:Dim);
  TestingNO=size(test_x,1);
  for pp=1:TestingNO
    %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
    actualvalue=My_FNN(size(test_x,2),HiddenNodes,len,Weights,Biases,test_x(pp,:));
    out=(output_classes==test_y(pp));
    if(sum((actualvalue-out').^2)==0)
      Rrate=Rrate+1;
    end
  end
        
  ClassificationRate=(Rrate/TestingNO)*100;
  disp(['Testing Classification rate = ', num2str(ClassificationRate)]);

% Draw the convergence curve
  hold on;      
  semilogy(ConvergenceCurve);
  title(['Testing Classification rate : ', num2str(ClassificationRate), '%']); 
  xlabel('Iteration');
  ylabel('MSE');
  box on
  grid on
  axis tight
  hold off;