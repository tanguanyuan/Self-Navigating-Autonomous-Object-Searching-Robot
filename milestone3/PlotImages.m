clear all; close all; clc; 

% Converting labels from Roboflow to labels in repository
pic_dir = 'images_new';
label_dir = 'labels_new';
pic_files = dir(sprintf('%s/*.png', pic_dir));
label_files = dir(sprintf('%s/*.txt', label_dir));

for i = 1:length(label_files)
    im = imread(sprintf('%s/%s', pic_dir, pic_files(i).name));
    figure(1);
    imshow(im);
    hold on; 
    txt = fileread(sprintf('%s/%s', label_dir, label_files(i).name));
    bbox = textscan(txt, '%f %f %f %f %f');
    w = size(im, 2);
    h = size(im, 1);
    x_center = bbox{2}*w;
    y_center = bbox{3}*h;
    width = bbox{4}*w;
    height = bbox{5}*h;
    x = x_center - width/2;
    y = y_center - height/2;
    rectangle('Position', [x, y, width, height])

    pause();
end