clear all;
clc;
% import data
im2 = imread('view1.png');
gt_disparity = imread('disp1.png');
gt_disparity = gt_disparity';
gt_disparity = double(gt_disparity);
% compare the results
sppt_mod = importdata('Support_Points_mod.txt');
sppt_ori = importdata('Support_Points_ori.txt');
sppt_mod(:,3) = sppt_mod(:,3)*4;
sppt_ori(:,3) = sppt_ori(:,3)*4;
scatter(sppt_mod(:,1),sppt_mod(:,2));
sppt_mod = double(sppt_mod);
sppt_ori = double(sppt_ori);
subplot(1,2,1);
imshow(im2);
hold on; 
scatter(sppt_mod(:,1),sppt_mod(:,2),20,'o', 'MarkerEdgeColor', [1 0 0]);
title('Support Match after Modification');
subplot(1,2,2);
imshow(im2);
hold on;
scatter(sppt_ori(:,1),sppt_ori(:,2), 20, 'o', 'MarkerEdgeColor',[1 0 0],...
    'LineWidth',0.5);
title('Original Support Match Results');
%calculate the error
err_mod = [];
sum_err_mod = 0;
for i = 1:length(sppt_mod)
    u = sppt_mod(i,1);
    v = sppt_mod(i,2);
    disp_cal_mod = sppt_mod(i,3);
    disp_gt = gt_disparity(u,v);
    if (disp_gt ~= 0)
        error_sqr = (disp_cal_mod-disp_gt)*(disp_cal_mod-disp_gt);
        err_mod = [err_mod; error_sqr];
        sum_err_mod = sum_err_mod + error_sqr;        
    end

end
RMSE_mod = sqrt(sum_err_mod/length(sppt_mod));
err_ori = [];
sum_err_ori = 0;
for j = 1:length(sppt_ori)
    u = sppt_ori(j,1);
    v = sppt_ori(j,2);
    disp_cal_ori = sppt_ori(j,3);
    disp_gt = gt_disparity(u,v);
    if (disp_gt ~= 0) 
        error_sqr = (disp_cal_ori-disp_gt)*(disp_cal_ori-disp_gt);
        err_ori = [err_ori; error_sqr];
        sum_err_ori = sum_err_ori + error_sqr;
    end
end
RMSE_ori = sqrt(sum_err_ori/length(sppt_ori));



