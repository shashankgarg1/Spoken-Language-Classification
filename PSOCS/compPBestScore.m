function pBestScore=compPBestScore(pBest, HiddenNodes, Dim, TrainingNO, X, Y,len,output_classes)
  columns=size(X,2);
  use=columns+len;
  weight_limit=use*HiddenNodes;
  %use=7;
  Weights=zeros(1,weight_limit);
  for ww=1:(weight_limit)
    Weights(ww)=pBest(ww);
  end

  Biases=zeros(1,Dim-weight_limit);
  for bb=weight_limit+1:Dim
    Biases(bb-(weight_limit))=pBest(bb);
  end
  
  fitness=0;
  for pp=1:TrainingNO
    %actualvalue=My_FNN(4,HiddenNodes,3,Weights,Biases,I2(pp,:));
    actualvalue=My_FNN(columns,HiddenNodes,len,Weights,Biases,X(pp,:));
    out=(output_classes==Y(pp));
    fitness=fitness+sum((actualvalue-out').^2);
  end
  fitness=fitness/TrainingNO;
  pBestScore=fitness;
end