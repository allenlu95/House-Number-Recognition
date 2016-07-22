% Classify House Numbers with Neural Network
This matlab script includes all necessary steps to pre-process data,
train neural network, and show results of classification

chop is the number of columns of pixels chopped off at left/right sides in each image
chop = 6;
if perform histogram equalization, change doHistogram to 1
doHistogram = 1;
if perform black-and-white thresholding, change doBlackWhite to 1
doBlackWhite = 0;
n is the Neural Network Hidden Layer size
n = 100;
dims is the dimension of the chopped image
dims = [32 32-2*chop];
load the data train.mat first
load train_32x32.mat
y = y';
perform chopping on images
X = X(:,chop+1:32-chop,:,:);
get dimensions of the images
NOTE: w here is actually the height, h is actually the weight
I made a mistake and don't want to break too many codes later
[w,h,~,N] = size(X);
d = w*h;

convert X(color images) to Y(gray-scale)
Y = squeeze( 0.2989*X(:,:,1,:) + 0.5870*X(:,:,2,:) + 0.1140*X(:,:,3,:) );

do Black-and-White thresholding
if doBlackWhite == 1
    ave1 = mean(Y, 2);
    ave2 = median(ave1, 1);
    for i = 1 : N
        for k = 1 : h
            for j = 1 : w
                if Y(j,k,i) >= ave2(i);
                    Y(j,k,i) = 255;
                else
                    Y(j,k,i) = 0;
                end
            end
        end
    end
end

perform histogram equalization on gray-scaled images
if doHistogram == 1
    for i = 1 : N
        Y(:,:,i) = histeq(Y(:,:,i));
    end
end

change class labels into required format by neural network
y(i) represents the label of the ith image
class(:,i) is a vector of all zeros except a 1 on the jth row
where j is the class label
class = zeros(10,N);
for i = 1 : 10
    class(i,:) = ~(y - i);
end

change images into required format by neural network
Y is 32*32*N grayscale image data
data is 1024*N grayscale image data
data = zeros(d,N);
for i = 1 : h
    data((w*(i-1)+1):(w*i),:) = Y(:,i,:); 
end

train the pattern network
net = patternnet(n);
[net,tr] = train(net, data, class);

clear variables and load testing images
clearvars -except net chop dims doHistogram doBlackWhite
load test_32x32.mat
y = y';
perform chopping on images
X = X(:,chop+1:32-chop,:,:);
get images dimensions
NOTE: w here is actually the height, h is actually the weight
I made a mistake and don't want to break too many codes later
[w,h,~,N] = size(X);
d = w*h;


convert X(color images) to Y(gray-scale)
Y = squeeze( 0.2989*X(:,:,1,:) + 0.5870*X(:,:,2,:) + 0.1140*X(:,:,3,:) );

do Black-and-White thresholding
if doBlackWhite == 1
    ave1 = mean(Y, 2);
    ave2 = median(ave1, 1);
    for i = 1 : N
        for k = 1 : h
            for j = 1 : w
                if Y(j,k,i) >= ave2(i);
                    Y(j,k,i) = 255;
                else
                    Y(j,k,i) = 0;
                end
            end
        end
    end
end

perform histogram equalization on gray-scaled image
if doHistogram == 1
    for i = 1 : N
        Y(:,:,i) = histeq(Y(:,:,i));
    end
end

convert images into required format by Neural Network
data = zeros(d,N);
for i = 1 : h
    data((w*(i-1)+1):(w*i),:) = Y(:,i,:); 
end

get classification result
classification = net(data);
the output of neural network is a probabilistic distribution
instead, we convert them to a single classification
[~, result] = max(classification,[],1);

count the number of errors
error = 0;
for i = 1 : N
    if norm(y(:,i) - result(:,i)) ~= 0
        error = error + 1;
    end
end
error_rate = error / N;