function Y1_3 = Eigenvalue(curPulse)%�ú������ڼ�������ֵ

    a = max(curPulse); %1��ֵ
    b = min(curPulse); %1��ֵ
    c = lyapunov_wolf(curPulse); %1��ֵ��������ŵ��ָ��
    f = mean((curPulse-mean(curPulse)).^2); %���׾�
    g = mean((curPulse-mean(curPulse)).^3); %���׾�
    h = mean((curPulse-mean(curPulse)).^4); %�Ľ׾�
    d = cum3x(curPulse,curPulse,curPulse); %1��ֵ������ͳ���� 
    e = ApproximateEntropy(2,0.2*std(curPulse),curPulse);  %1��ֵ�������أ�Ƕ��ά����ʱ���ӳ١�ƽ������
    sample_entropy_value = sample_entropy(curPulse, 2);    %1��ֵ������������
    [fd, x] = fractal_dimension(curPulse);  %1��ֵ���������ά����fd�Ƿ���ά��
    area_under_pulse = trapz(curPulse);    % �������������ݵ����

    % �������������ݵĵ���
    derivative_pulse = diff(curPulse);
    zero_derivative_indices = find(derivative_pulse == 0);
    num_zero_derivative_points = length(zero_derivative_indices);

    % ��������ֵ�ֽ� %ѡ��ǰ10������ֵ���õ�10��ֵ
    curPulse1 = reshape(curPulse, 5, 12);
    [U, S, V] = svd(curPulse1);
    num_singular_values = 10;
    num_singular_values = min(num_singular_values, min(size(S)));
    singular_values = diag(S(1:num_singular_values, 1:num_singular_values));
    if length(singular_values) < 5
        singular_values = [singular_values; zeros(5 - length(singular_values), 1)];
    end
    singular_values = singular_values(:)';
  

    % ִ��ϣ������-�Ʊ任��% ��ȡǰ4��IMF�ľ�ֵ�ͷ����Ϊ����ֵ
    imf = emd(curPulse); % ����ģ̬�ֽ�
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


