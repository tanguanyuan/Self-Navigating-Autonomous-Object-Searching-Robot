clear all; close all; clc; 

% Converting labels from Roboflow to labels in repository

new_labels = [5 2 4 3 1];

load_label_dir = 'labels';
save_label_dir = 'labels_new';
load_images_dir = 'images';
save_images_dir = 'images_new';
label_files = dir(sprintf('%s/*.txt', load_label_dir));
images_files = dir(sprintf('%s/*.jpg', load_images_dir));

index = 20001;
for i = [1:2:length(label_files), 2:2:length(label_files)]
    % save label
    txt = fileread(sprintf('%s/%s', load_label_dir, label_files(i).name));
    old_label = str2double(txt(1));
    new_label = new_labels(old_label+1);
    newtxt = txt;
    newtxt(1) = num2str(new_label);
    name = sprintf('images_%d', index);
    save_name = sprintf('%s/%s%s', save_label_dir, name, '.txt');
    save_file = fopen(save_name, 'w');
    fprintf(save_file, newtxt);
    fclose(save_file);

    % save images
    im = imread(sprintf('%s/%s', load_images_dir', images_files(i).name));
    write_name = sprintf('%s/%s%s', save_images_dir, name, '.png');
    imwrite(im, write_name);

    index = index + 1;
end