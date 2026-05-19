%% TEST_FRUIT_DETECTION - Test fruit detection with synthetic or real images
% Creates synthetic test images to validate the fruit detector
% before using real camera images.

clear; clc; close all;
addpath('../matlab_code');

fprintf('=== Fruit Detection Test ===\n\n');

%% ---- Create Synthetic Test Image ----
% Simulates a 640x480 image with colored circles representing fruits
img = uint8(180 * ones(480, 640, 3));  % Light gray background (table)

% Draw strawberries (red, elongated)
img = insertShape(img, 'FilledCircle', [200, 150, 25], 'Color', [200 30 30], 'Opacity', 1);
img = insertShape(img, 'FilledCircle', [220, 160, 20], 'Color', [180 20 20], 'Opacity', 1);

% Draw tomatoes (bright red, round)
img = insertShape(img, 'FilledCircle', [400, 200, 30], 'Color', [230 40 40], 'Opacity', 1);
img = insertShape(img, 'FilledCircle', [350, 300, 28], 'Color', [220 35 35], 'Opacity', 1);

% Draw bananas (yellow, elongated)
img = insertShape(img, 'FilledCircle', [150, 350, 22], 'Color', [240 220 50], 'Opacity', 1);
img = insertShape(img, 'FilledCircle', [500, 250, 24], 'Color', [230 210 40], 'Opacity', 1);
img = insertShape(img, 'FilledCircle', [300, 400, 20], 'Color', [245 225 55], 'Opacity', 1);

% Add some noise for realism
noise = uint8(randn(480, 640, 3) * 5 + 128) - 128;
img = uint8(double(img) + double(noise));

%% ---- Run Detection ----
fprintf('Running fruit detection...\n');
[types, centroids, bboxes, confs] = fruitDetector(img, []);

fprintf('\nDetected %d fruits:\n', length(types));
for i = 1:length(types)
    fprintf('  %d. %s at pixel [%.0f, %.0f], confidence=%.2f\n', ...
        i, types{i}, centroids(i,1), centroids(i,2), confs(i));
end

%% ---- Visualize ----
figure('Name', 'Fruit Detection Test', 'Position', [100, 100, 800, 600]);
imshow(img);
hold on;

color_map = containers.Map({'strawberry','tomato','banana'}, ...
    {[1 0 0], [1 0 1], [1 1 0]});

for i = 1:length(types)
    bb = bboxes(i,:);
    c = color_map(types{i});
    rectangle('Position', bb, 'EdgeColor', c, 'LineWidth', 2);
    text(bb(1), bb(2)-10, sprintf('%s (%.0f%%)', types{i}, confs(i)*100), ...
        'Color', c, 'FontSize', 10, 'FontWeight', 'bold', ...
        'BackgroundColor', [0 0 0 0.5]);
end
title('Fruit Detection Results (Synthetic Image)');

saveas(gcf, '../figures/detection_test.png');
fprintf('\nFigure saved to ../figures/detection_test.png\n');
fprintf('=== Detection Test Complete ===\n');
