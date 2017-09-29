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
  TrainingNO=size(train_x,1);
  use=columns+len;
  noP=30;           %Number of particles
  Max_iteration=500;%Maximum number of iterations
  
  ConvergenceCurve=zeros(1,Max_iteration); %Convergence vector
  
    % Discovery rate of alien eggs/solutions
    pa=0.25;

    % Change this if you want to get better results
    % Tolerance
    Tol=1.0e-5;
    % Simple bounds of the search domain
    % Lower bounds
    nd=Dim;
    n=noP;
    Lb=-1.0*ones(1,nd); 
    % Upper bounds
    Ub=1.0*ones(1,nd);
    
    % Random initial solutions
    nest=zeros(noP,Dim);
    for i=1:noP,
        nest(i,:)=Lb+(Ub-Lb).*randn(size(Lb));
    end
  
    weight_limit=use*HiddenNodes;
    fitness=10^10*ones(n,1);
    [fmin,bestnest,nest,fitness]=get_best_nest(nest,nest,fitness,train_x,train_y,columns,HiddenNodes,len,weight_limit,output_classes,use,Dim,TrainingNO);

    N_iter=0;
    % Starting iterations
%    while (fmin>Tol)
     while(N_iter<=25000)

        % Generate new solutions (but keep the current best)
        new_nest=get_cuckoos(nest,bestnest,Lb,Ub);   
        [fnew,best,nest,fitness]=get_best_nest(nest,new_nest,fitness,train_x,train_y,columns,HiddenNodes,len,weight_limit,output_classes,use,Dim,TrainingNO);
        % Update the counter
        N_iter=N_iter+1; 
        % Discovery and randomization
        new_nest=empty_nests(nest,Lb,Ub,pa) ;
    
        % Evaluate this set of solutions
        [fnew,best,nest,fitness]=get_best_nest(nest,new_nest,fitness,train_x,train_y,columns,HiddenNodes,len,weight_limit,output_classes,use,Dim,TrainingNO);
        % Update the counter again
        %N_iter=N_iter+1;
        % Find the best objective so far  
        if fnew<fmin,
            fmin=fnew;
            bestnest=best;
        end
        ConvergenceCurve(1,N_iter)=fmin;
        disp(['CS is training FNN (Iteration = ', num2str(N_iter),' ,MSE = ', num2str(fmin),')']) 
    end %% End of iterations

%% Post-optimization processing
%% Display all the nests
disp(strcat('Total number of iterations=',num2str(N_iter)));
fmin
bestnest

% Calculate the classification for training
  Rrate=0;
  Weights=bestnest(1:weight_limit);
  Biases=bestnest(weight_limit+1:Dim);
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
  Weights=bestnest(1:weight_limit);
  Biases=bestnest(weight_limit+1:Dim);
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