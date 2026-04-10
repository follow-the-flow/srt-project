function [phi, phi_optimal] = qarmInverseKinematics(p, gamma, phi_prev)
%% QARM_INVERSE_KINEMATICS
% 基于 Quanser QArm 几何结构与 DH 参数的逆运动学实现

%% 1. 机械臂物理参数 (单位: 米)
L1 = 0.1400;
L2 = 0.3500;
L3 = 0.0500;
L4 = 0.2500;
L5 = 0.1500;

% 几何辅助参数
beta = atan2(L3, L2);                 % β = tan⁻¹(L3/L2)
lambda1 = L1;
lambda2 = sqrt(L2^2 + L3^2);
lambda3 = sqrt(L4^2 + L5^2);

%% 2. 初始化
theta = zeros(4, 4);
phi = zeros(4, 4);
x = p(1);
y = p(2);
z = p(3);

%% 3. 求解关节 2 和 关节 3 (几何法)
A = lambda2;
C = lambda3;
H = lambda1 - z;

% 计算两个可能的 D 值（对应基座旋转 0° 或 180°）
D1 = sqrt(x^2 + y^2);  
D2 = -sqrt(x^2 + y^2); 

% 计算 theta3 的四组解 (基于余弦定理变形)
F1 = (A^2 + C^2 - D1^2 - H^2) / (2 * A);
theta(3,1) = 2*atan2( C + sqrt(C^2 - F1^2), F1 );
theta(3,2) = 2*atan2( C - sqrt(C^2 - F1^2), F1 );

F2 = (A^2 + C^2 - D2^2 - H^2) / (2 * A);
theta(3,3) = 2*atan2( C + sqrt(C^2 - F2^2), F2 );
theta(3,4) = 2*atan2( C - sqrt(C^2 - F2^2), F2 );

% 计算 theta2 的四组解
for i = 1:4
    curr_D = D1; if i > 2, curr_D = D2; end
    M = A + C * cos(theta(3,i));
    % 注意：根据 DH 坐标定义，此处 N 对应 sin 项
    N = C * sin(theta(3,i));
    theta(2,i) = atan2(M*H + N*curr_D, M*curr_D - N*H);
end

%% 4. 求解关节 1
theta(1,1) = atan2(y, x);
theta(1,2) = atan2(y, x);
theta(1,3) = atan2(y, x) + pi;
theta(1,4) = atan2(y, x) + pi;

%% 5. 映射回物理关节角 phi
% 这里的逻辑必须与正运动学定义的映射关系严格互逆
phi(1,:) = theta(1,:);
phi(2,:) = theta(2,:) - pi/2 + beta;
phi(3,:) = theta(3,:) - beta;
phi(4,:) = theta(4,:) + gamma; 

% 角度归一化到 [-pi, pi] 范围内
phi = mod(phi + pi, 2*pi) - pi;

%% 6. 选择最优解 (phi_optimal)
% 策略：在满足关节物理限位的前提下，选择离当前位置 phi_prev 距离最近的解
phi_optimal = [0; 0; 0; 0]; 
min_dist = inf;

for i = 1:4
    curr_phi = phi(:,i);
    % 检查该解是否在物理限位范围内
    if ~check_joint_limits(curr_phi)
        % 计算欧几里得距离（最短路径原则）
        dist = norm(curr_phi - phi_prev); 
        if dist < min_dist
            min_dist = dist;
            phi_optimal = curr_phi;
        end
    end
end

% 兜底方案：如果所有解都超出物理限位，则返回安全位置（如 Home 点）
if (check_joint_limits(phi_optimal))
    phi_optimal = [0; 0; 0; 0];
end

end

%% 辅助函数：关节限位检查
function flag = check_joint_limits(phi)
    flag = 0;
    % 根据 QArm 硬件规格定义的弧度限位
    if phi(1) > 170*pi/180 || phi(1) < -170*pi/180 ...
        || phi(2) > 80*pi/180 || phi(2) < -80*pi/180 ...
        || phi(3) > 75*pi/180 || phi(3) < -95*pi/180 ...
        || phi(4) > 160*pi/180 || phi(4) < -160*pi/180
        flag = 1;
    end
end