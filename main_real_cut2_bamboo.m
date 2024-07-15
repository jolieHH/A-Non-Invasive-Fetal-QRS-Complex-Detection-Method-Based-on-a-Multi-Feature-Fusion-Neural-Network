clc;
clear all;
addpath('D:\博士生\学习\论文发表\测孕IEEEtrans论文封装\bamboo孕妇matlab和python');
addpath('D:\博士生\学习\论文发表\是否包含集群二分类\set-a\varanini');
addpath('D:\博士生\学习\论文发表\是否包含集群二分类\set-a');
n_points_to_plot = 60000  % 源数据的总点数
segment_length_2000 = 500;  % 每个段落的点数
n_channels = 4;
max_AAA = 5;%75;  % 最大的AAA值

% 循环处理所有文件
for i = 1:max_AAA
    file_name = sprintf('r%02d', i);
    file_name_all = strcat(file_name, '.dat');
    if exist(file_name_all, 'file')
        [signals_before, header] = rdsamp(file_name);
        ann = rdann(file_name, 'fqrs'); % 读取注释文件 
        
        signals_before = signals_before(1:n_points_to_plot, :);
        ann = ann(ann <= n_points_to_plot);
        
        % 初始化类别和数据存储
        data_0 = []; % 0类别数据，不包含qrs
        data_1 = []; % 1类别数据，包含qrs

        % 初始化通道的样本熵矩阵
        sample_entropy_channel = zeros(1, n_channels);
        for channel = 1:n_channels
                signal_before = signals_before(:, channel);
                [signals, fetal_QRSAnn_est, QT_Interval] = physionet2013(signal_before);
                signal = DenoiseSig(signals);
                signal = (signal - min(signal)) / (max(signal) - min(signal));
                signal_before = DenoiseSig(signal_before);
                signal_before = (signal_before - min(signal_before)) / (max(signal_before) - min(signal_before));
                
                % 初始化样本熵的数组
                sample_entropy_array = zeros(1, 3);
        
                % 随机抽取段落进行子采样
                for seg = 1:10
                    start_index = randi(n_points_to_plot - segment_length_2000 + 1);
                    end_index = start_index + segment_length_2000 - 1;

                    % 获取当前段落
                    signal_2000 = signal(start_index:end_index, 1);
              
                    % 计算样本熵
                    m = 2; % 嵌入维度
                    std_percentage = 520; % 百分比法中的标准差百分比
                    r = std(signal_2000) * (std_percentage / 100); % 匹配间隔
                    sample_entropy = sample_entropy2(signal_2000, m, r);
                    
                    % 将样本熵值保存到数组中
                    sample_entropy_array(seg) = sample_entropy;
                end
        
                % 求均值并保存到通道的样本熵矩阵中
                sample_entropy_channel(channel) = mean(sample_entropy_array);
                if sample_entropy_channel(channel) > 0.55/100
                     for ii = 1:length(ann)
                        ann_point = ann(ii);
                        segment_length = 60;%40是-24：-15
                        % 处理包含ann标记点的数据
                        for j = -34:2:-25
                            start_idx = ann_point + j;
                            end_idx = start_idx + segment_length - 1;
                
                            if start_idx > 0 && end_idx <= length(signal_before)
                                segment = signal_before(start_idx:end_idx);
                                data_1 = [data_1; segment'];
                            end
                        end
                
                        % 处理不包含ann标记点的数据
                        for j = 1:2:10
                            start_idx = ann_point + 30 + (j - 1) * segment_length;
                            end_idx = ann_point + 30 + segment_length + (j - 1) * segment_length - 1;
                
                            if start_idx > 0 && end_idx < length(signal_before)
                                segment = signal_before(start_idx:end_idx);
                
                                % 检查是否包含其他ann标记点
                                if isempty(intersect(segment, ann))
                                    data_0 = [data_0; segment'];
                                end
                            end
                        end
                    end
                end
        end

% signal_before = signals_before(:, 1);
% fqrs_indices = ann;
% figure;
% n_points_to_plot = 6000;
% 
% % Subplot for signal_before
% subplot(2, 1, 1);
% plot(signal_before(1:n_points_to_plot));
% hold on;
% plot(fqrs_indices(fqrs_indices <= n_points_to_plot), signal_before(fqrs_indices(fqrs_indices <= n_points_to_plot)), 'ro'); % 画出 fqrs 的位置
% title('处理前的信号 - 通道1');
% xlabel('时间');
% ylabel('幅度');
% 
% % Subplot for signal
% subplot(2, 1, 2);
% plot(signal(1:n_points_to_plot));
% hold on;
% plot(fqrs_indices(fqrs_indices <= n_points_to_plot), signal(fqrs_indices(fqrs_indices <= n_points_to_plot)), 'ro'); % 画出 fqrs 的位置
% title('处理后的信号 - 通道1');
% xlabel('时间');
% ylabel('幅度');
% % Adjusting figure properties for better visualization
% set(gcf, 'Position', [100, 100, 800, 600]); % Adjust figure size and position


% % 假设 data_1 和 data_0 已经被定义
% 
% % 选择两行数据进行绘制
% data_1_samples = data_1(1:2, :);
% data_0_samples = data_0(1:2, :);
% 
% % 生成 x 轴坐标（假设数据点的时间均匀分布）
% x_axis = 1:50;
% 
% % 绘制 data_1 中的两行数据
% figure;
% subplot(2, 1, 1);
% plot(x_axis, data_1_samples(1, :), '-o', 'DisplayName', 'Data 1 - Row 1');
% hold on;
% plot(x_axis, data_1_samples(2, :), '-s', 'DisplayName', 'Data 1 - Row 2');
% title('示意图 - data_1');
% legend;
% hold off;
% 
% % 绘制 data_0 中的两行数据
% subplot(2, 1, 2);
% plot(x_axis, data_0_samples(1, :), '-o', 'DisplayName', 'Data 0 - Row 1');
% hold on;
% plot(x_axis, data_0_samples(2, :), '-s', 'DisplayName', 'Data 0 - Row 2');
% title('示意图 - data_0');
% legend;
% hold off;


%###
        % 定义窗口大小和初始步进大小（以样本点为单位）       
%         window_size = 100;%150;
%         step_size_no_qrs = 80;%50;
%         step_size_qrs = 12;%30;
%        
%         j = 1; % 初始化起始位置
%         while j <= length(signals)-window_size+1
%             window_data = signals(j:j+window_size-1, :);
%             ann_window = ann(ann >= j & ann < j+window_size);
%     
%             if isempty(ann_window)
%                 data_0 = [data_0; window_data(:)'];
%                 j = j + step_size_no_qrs; % 没有QRS波群，向后移动40毫秒
%             else
%                 data_1 = [data_1; window_data(:)'];
%                 j = j + step_size_qrs; % 有QRS波群，向后移动10毫秒
%             end
%         end
 %###   



%         % 对每一行进行翻转
%         flipped_data_0 = 1 - data_0;       
%         % 将翻转后的数据与原始数据拼接，以实现数据量翻倍
%         augmented_data_0 = [data_0; flipped_data_0];
%         % 至此，augmented_data_0 包含了对原始数据进行每一行上下翻转（以1为中线）后的增强数据

        n_train = size(data_0,1);%返回行数,data_0是一条长数据切割成的归类为0的所有，100列
        for m = 1: n_train
            curPulse = data_0(m,:);
            curPulse(isnan(curPulse(1,:))==1)=[0];
            curPulse(isinf(curPulse(1,:))==1)=[1];
%             plot(curPulse);%hold on;
            Y0_1 = curPulse; %时域，1*80
            Y0_1 = reshape(Y0_1, segment_length, 1)';%2*40
            Y0_2 = abs(fft(curPulse));%频域信号，1*80
            Y0_2 = Y0_2(1:segment_length);%1*40
            Y0_3 = Eigenvalue(curPulse);%特征值信号，1*40
            Y0_4 = cwt(curPulse); %plot(Y1);hold on;       %
            %size(Y0_4)
            Y0_4 = imresize(Y0_4,[24,segment_length],'bilinear');   
            Y0_4 = abs(Y0_4);
            Y0_4 = Y0_4/max(max(Y0_4))*2-1;%时频域 34*40     mesh(Y1);
            Y0 = [Y0_1;Y0_2;Y0_3;Y0_4];

            if any(isnan(Y0(:)))
                disp('Y0 contains NaN values. Skipping...');
            else
                [Y_size1, Y_size2] = size(Y0);   
                fi = ['D:\博士生\学习\论文发表\是否包含集群二分类\datanew\0_', num2str(i), '.txt'];
                fip = fopen(fi, 'a'); % 打开或创建要写入的新文件。追加数据到文件末尾。
                for ii = 1:Y_size1  
                    for j = 1:Y_size2  
                        fprintf(fip, '%g\t', Y0(ii, j));  % 插入制表符和按照小数罗列
                    end
                    fprintf(fip, '\r\n'); % 回车 换行  
                end   
                fclose(fip);
            end
        end
    
%         % 对每一行进行翻转
%         flipped_data_1 = 1 - data_1;       
%         % 将翻转后的数据与原始数据拼接，以实现数据量翻倍
%         augmented_data_1 = [data_1; flipped_data_1];
%         % 至此，augmented_data_0 包含了对原始数据进行每一行上下翻转（以1为中线）后的增强数据

         n_train = size(data_1,1); % 返回行数,data_1是一条长数据切割成的归类为1的所有
         for n = 1: n_train
            curPulse = data_1(n,:);
            curPulse(isnan(curPulse(1,:))==1)=[0];
            curPulse(isinf(curPulse(1,:))==1)=[1];
            % plot(curPulse);%hold on;
            Y1_1 = curPulse; %时域，1*40
            Y1_1 = reshape(Y1_1, segment_length, 1)';%2*40
            Y1_2 = abs(fft(curPulse));%频域信号，1*40
            Y1_2 = Y1_2(1:segment_length);%1*40
            Y1_3 = Eigenvalue(curPulse);%特征值信号，1*40
            Y1_4 = cwt(curPulse); %plot(Y1);hold on;       %
            %size(Y1_4)
            Y1_4 = imresize(Y1_4,[24,segment_length],'bilinear');   
            Y1_4 = abs(Y1_4);
            Y1_4 = Y1_4/max(max(Y1_4))*2-1;%时频域 34*40     mesh(Y1);
            Y1 = [Y1_1;Y1_2;Y1_3;Y1_4];
            
            [Y_size1, Y_size2] = size(Y1);   
            fi = ['D:\博士生\学习\论文发表\是否包含集群二分类\datanew\1_', num2str(i), '.txt'];
            fip = fopen(fi, 'a'); % 打开或创建要写入的新文件。追加数据到文件末尾。
            
            % 检查Y1是否包含NaN值
            if any(isnan(Y1(:)))
                disp('Y1 contains NaN values. Skipping...');
            else
                for ii = 1:Y_size1  
                    for j = 1:Y_size2  
                        fprintf(fip, '%g\t', Y1(ii, j));  % 插入制表符和按照小数罗列
                    end
                    fprintf(fip, '\r\n'); % 回车 换行  
                end   
            end
            fclose(fip);
         end
    else
        disp(['文件 ' file_name ' 不存在，跳过处理']);
    end 
end
fclose('all');
