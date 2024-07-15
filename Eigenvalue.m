function Y1_3 = Eigenvalue(curPulse)%该函数用于计算特征值

    a = max(curPulse); %1个值
    b = min(curPulse); %1个值
    c = lyapunov_wolf(curPulse); %1个值，李雅普诺夫指数
    f = mean((curPulse-mean(curPulse)).^2); %二阶矩
    g = mean((curPulse-mean(curPulse)).^3); %三阶矩
    h = mean((curPulse-mean(curPulse)).^4); %四阶矩
    d = cum3x(curPulse,curPulse,curPulse); %1个值，三阶统计量 
    e = ApproximateEntropy(2,0.2*std(curPulse),curPulse);  %1个值，近似熵，嵌入维数、时间延迟、平均周期
    sample_entropy_value = sample_entropy(curPulse, 2);    %1个值，计算样本熵
    [fd, x] = fractal_dimension(curPulse);  %1个值，计算分形维数，fd是分形维数
    area_under_pulse = trapz(curPulse);    % 计算脉搏波数据的面积

    % 计算脉搏波数据的导数
    derivative_pulse = diff(curPulse);
    zero_derivative_indices = find(derivative_pulse == 0);
    num_zero_derivative_points = length(zero_derivative_indices);

    % 进行奇异值分解 %选择前10个奇异值，得到10个值
    curPulse1 = reshape(curPulse, 5, 12);
    [U, S, V] = svd(curPulse1);
    num_singular_values = 10;
    num_singular_values = min(num_singular_values, min(size(S)));
    singular_values = diag(S(1:num_singular_values, 1:num_singular_values));
    if length(singular_values) < 5
        singular_values = [singular_values; zeros(5 - length(singular_values), 1)];
    end
    singular_values = singular_values(:)';
  

    % 执行希尔伯特-黄变换，% 提取前4个IMF的均值和方差，作为特征值
    imf = emd(curPulse); % 经验模态分解
    features = zeros(4, 2);
    for i = 1:min(4, size(imf, 2))
        features(i, 1) = mean(imf(:, i));
        features(i, 2) = var(imf(:, i));
    end
    flattened_features = reshape(features', 1, []);
    if length(flattened_features) < 5
         flattened_features = [flattened_features; zeros(5 - length(flattened_features), 1)];
    end

    mm = [a,b,c,f,g,h,d,e,sample_entropy_value,fd, area_under_pulse, num_zero_derivative_points,singular_values,flattened_features];
    Y1_3 = padarray(mm, [0, 60-length(mm)], 0, 'post');
end


