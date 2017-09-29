function [fmin,best,nest,fitness]=get_best_nest(nest,newnest,fitness,X,Y,columns,HiddenNodes,len,weight_limit,output_classes,use,Dim,TrainingNO)
    Weights=zeros(1,use*HiddenNodes);
    Biases=zeros(1,Dim-use*HiddenNodes);
% Evaluating all new solutions
    for j=1:size(nest,1)
        for ww=1:(weight_limit)
            Weights(ww)=newnest(j,ww);
        end
      
        for bb=weight_limit+1:Dim
            Biases(bb-(weight_limit))=newnest(j,bb);
        end
        
        fnew=0;
        for pp=1:TrainingNO
            %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
            actualvalue=My_FNN(columns,HiddenNodes,len,Weights,Biases,X(pp,:));
            out=(output_classes==Y(pp));
            fnew=fnew+sum((actualvalue-out').^2);
        end
        fnew=fnew/TrainingNO;
        %fnew=fobj(newnest(j,:));
        if fnew<=fitness(j),
            fitness(j)=fnew;
            nest(j,:)=newnest(j,:);
        end
    end
    % Find the current best
    [fmin,K]=min(fitness) ;
    best=nest(K,:);
end