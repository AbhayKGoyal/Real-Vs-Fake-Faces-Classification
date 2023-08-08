clc
clear all 
clc

%%From the Real & Fake face images, the vectorization has been carried out and 1D
%%vector for an image has been obtained. Note, the same images used in
%%Real / Fake Face classification has been used in this project as well.

%RBY-array size 20*20*3
% load X.mat %both your Real and Fake trainset has been saved in this variable
% load y.mat %Real-1 Fake-0 target variable
% load X_test.mat %test variable
% load Real_train.mat
% load Fake_train.mat
% load Real_test.mat
% load Fake_test.mat

[training_data_1, m1]=process_data('dataset\train\fake',20,20,3);
y1=ones(m1,1);

x11=size(y1)
x1=length(y1)
pause;
[training_data_2, m2]=process_data('dataset\train\real',20,20,3);
y2=zeros(m2,1);
x22=size(y2)
x2=length(y2)
pause;
training_data=[training_data_1; training_data_2];
y=[y1; y2];
x3=size(y)
x33=length(y)
x4=size(training_data)
x44=length(training_data)

X=training_data;
[test_data_1 s1]=process_data('dataset\test\fake',20,20,3)
[test_data_3 s2]=process_data('dataset\test\real',20,20,3)
test_data = [test_data_1 ; test_data_3];
yactual = [ones(s1,1);zeros(s2,1)];
%%%Developing model using SVM
SVMModel = fitcsvm(X,y,'Standardize',true,'KernelFunction','rbf','KernelScale','auto');

P=predict(SVMModel,test_data);
% P4=predict(SVMModel,test_data_3);
 fprintf("svm\n");
F1 = f1(yactual,P)


k= 5;
fprintf("knn\n");
ypred=knn(X,y,k,test_data);
F1 = f1(yactual,ypred)


pause




[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];
y=y;
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

%%%%optimised value of theta
theta


% Compute accuracy on our training set
[m1, n1] = size(test_data);
Xt=[ones(m1, 1) test_data];
p = predict_1(theta, Xt);
 [Cmatrix,ACC,P,R,F1]=confusionmatrix(p,yactual);

 fprintf('F1 Score using logistic regression without regularization\n');
 Cmatrix
 F1
%%%%%%%%%%%%%%%%%%Logistic regression with regularization
% Set regularization parameter lambda to 1
lambda = 10;
initial_theta = zeros(size(X, 2), 1);
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

%%%%optimised value of theta
theta;



% Compute accuracy on our training set
p = predict_1(theta, Xt);
 [Cmatrix,ACC,P,R,F1]=confusionmatrix(p,ytest);
 
 fprintf('F1 Score using logistic regression with regularization\n');
% Cmatrix
 F1
 
