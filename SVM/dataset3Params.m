function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
min_error = -1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for temp_C = candidate
    for temp_sigma = candidate
        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        predictions = svmPredict(model, Xval);
        error_this_time = mean(double(predictions ~= yval));
        if(min_error == -1 || error_this_time < min_error)
            min_error = error_this_time;
            C = temp_C;
            sigma = temp_sigma;
        end
    end
end

% =========================================================================

end
