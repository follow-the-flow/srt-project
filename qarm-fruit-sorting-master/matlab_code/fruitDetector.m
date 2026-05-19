function [fruit_type, centroid_px, bounding_box, confidence] = fruitDetector(rgb_image, depth_image)
%FRUITDETECTOR Detects and classifies fruits using color segmentation
%   Uses HSV color space to identify strawberries, bananas, and tomatoes
%   from the QArm's Intel RealSense D415 RGBD camera.
%
%   INPUT:  rgb_image   - HxWx3 uint8 RGB image from RealSense
%           depth_image - HxW uint16 or single depth map (mm)
%   OUTPUT: fruit_type   - Nx1 cell array: 'strawberry', 'banana', 'tomato'
%           centroid_px  - Nx2 pixel coordinates [row, col]
%           bounding_box - Nx4 [x, y, width, height] for each detection
%           confidence   - Nx1 confidence scores [0, 1]
%
%   Color ranges in HSV (tuned for typical lab lighting):
%     Strawberry: H=[0,10]U[170,180], S=[100,255], V=[80,255] (red/dark red)
%     Banana:     H=[20,35], S=[100,255], V=[120,255] (yellow)
%     Tomato:     H=[0,8], S=[150,255], V=[100,255] (bright red, rounder)
%
%   Discrimination between strawberry and tomato:
%     - Tomatoes are rounder (circularity > 0.7)
%     - Strawberries are more elongated (circularity < 0.7)
%     - Tomatoes tend to be brighter red with higher saturation

    if nargin < 2
        depth_image = [];
    end

    % Convert to HSV
    hsv = rgb2hsv(rgb_image);
    H = hsv(:,:,1) * 180;  % Scale to [0, 180] for OpenCV convention
    S = hsv(:,:,2) * 255;
    V = hsv(:,:,3) * 255;

    % ---- Banana detection (yellow) ----
    banana_mask = (H >= 20 & H <= 35) & (S >= 100) & (V >= 120);
    banana_mask = cleanMask(banana_mask, 500);

    % ---- Red detection (strawberry + tomato share red hue) ----
    red_mask = ((H >= 0 & H <= 10) | (H >= 170 & H <= 180)) & (S >= 100) & (V >= 80);
    red_mask = cleanMask(red_mask, 400);

    % Initialize outputs
    fruit_type = {};
    centroid_px = [];
    bounding_box = [];
    confidence = [];

    % ---- Process bananas ----
    [labels_b, n_b] = bwlabel(banana_mask);
    props_b = regionprops(labels_b, 'Centroid', 'BoundingBox', 'Area', 'Perimeter');
    for i = 1:n_b
        if props_b(i).Area > 300
            fruit_type{end+1,1} = 'banana';
            centroid_px(end+1,:) = props_b(i).Centroid([2,1]); % [row, col]
            bounding_box(end+1,:) = props_b(i).BoundingBox;
            % Confidence based on area and color purity
            conf = min(1.0, props_b(i).Area / 5000);
            confidence(end+1,1) = conf;
        end
    end

    % ---- Process red fruits (distinguish strawberry vs tomato) ----
    [labels_r, n_r] = bwlabel(red_mask);
    props_r = regionprops(labels_r, 'Centroid', 'BoundingBox', 'Area', 'Perimeter');
    for i = 1:n_r
        if props_r(i).Area > 300
            % Compute circularity: 4*pi*Area / Perimeter^2
            % Circle = 1.0, elongated < 0.7
            circ = 4 * pi * props_r(i).Area / (props_r(i).Perimeter^2 + eps);

            % Get mean saturation in this region
            region_mask = (labels_r == i);
            mean_sat = mean(S(region_mask));

            if circ > 0.65 && mean_sat > 140
                fruit_type{end+1,1} = 'tomato';
            else
                fruit_type{end+1,1} = 'strawberry';
            end

            centroid_px(end+1,:) = props_r(i).Centroid([2,1]);
            bounding_box(end+1,:) = props_r(i).BoundingBox;
            conf = min(1.0, props_r(i).Area / 4000);
            confidence(end+1,1) = conf;
        end
    end
end

function mask = cleanMask(mask, min_area)
%CLEANMASK Morphological cleanup of binary mask
    se = strel('disk', 5);
    mask = imopen(mask, se);
    mask = imclose(mask, se);
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, min_area);
end
