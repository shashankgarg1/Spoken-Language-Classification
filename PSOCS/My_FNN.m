function eoi=My_FNN(Ino,Hno,Ono,W,B,x)
  h=zeros(1,Hno);
  o=zeros(1,Ono);

  for i=1:Hno
    %for j=1:size(x,2)
    for j=1:Ino
      h(i)=h(i)+x(j)*W((j-1)*Hno+i);
    end
    h(i)=h(i)+B(i);
    h(i)=My_sigmoid(h(i));
  end

%k=3;

% computing softmax
  k=Ino-1;
  for i=1:Ono
    k=k+1;
    for j=1:Hno
      o(i)=o(i)+(h(j)*W(k*Hno+j));
    end
  end

  sume=0;
  eo=zeros(1,Ono);
  for i=1:Ono
    o(i)=o(i)+B(Hno+i);
    eo(i)=exp(o(i));
    sume=sume+eo(i);
  end
  
  for i=1:Ono
    eo(i)=eo(i)/sume;
  end
  eoi=(eo==max(eo));
end