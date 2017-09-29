function [X,Y]=read_data(str)
  dataset=csvread(str);
  len=size(dataset,1);
  
  %shuffling of dataset
  dataset=dataset(randperm(len),:);
  
  %separating X and Y
  X=dataset(:,2:size(dataset,2)-1);
  Y=dataset(:,size(dataset,2));
  
  %feature scaling
  normx=max(X)-min(X);
  normx=repmat(normx,[length(X) 1]);
  X=X./normx;
end