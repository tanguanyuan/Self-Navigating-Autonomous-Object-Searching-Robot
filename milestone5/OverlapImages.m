clear all; close all; clc;

figure(1);
for i = 0:0
    figure(i+1);
    im = imread(sprintf('pibot_dataset/img_%d.png', i));
    label = imread(sprintf('lab_output/pred_%d.png', i));
    label = imresize(label, [size(im,1), size(im, 2)]);
    imcont = imfuse(im, label2rgb(label));
    imshow(imcont);
end
