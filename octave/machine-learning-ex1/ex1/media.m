function media = media()
  
  X = [2,20;2,20;3,40;3,50]
  
  X_norm = zeros(size(X,1),size(X,2))
  
  M = zeros(1,size(X,2))
  
  M = mean(X)
  STD = std(X)
  
  sizes = size(M,2)
  
  for i = 1:size(M,2)
    X(:,i) = X(:,i) - M(i)
    X(:,i) = X(:,i) / STD(i)
  end;
     
endfunction
