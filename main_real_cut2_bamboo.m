clc;
clear all;
addpath('D:\��ʿ��\ѧϰ\���ķ���\����IEEEtrans���ķ�װ\bamboo�и�matlab��python');
addpath('D:\��ʿ��\ѧϰ\���ķ���\�Ƿ������Ⱥ������\set-a\varanini');
addpath('D:\��ʿ��\ѧϰ\���ķ���\�Ƿ������Ⱥ������\set-a');
n_points_to_plot = 60000  % Դ���ݵ��ܵ���
segment_length_2000 = 500;  % ÿ������ĵ���
n_channels = 4;
max_AAA = 5;%75;  % ����AAAֵ

% ѭ�����������ļ�
for i = 1:max_AAA
    file_name = sprintf('r%02d', i);
    file_name_all = strcat(file_name, '.dat');
    if exist(file_name_all, 'file')
        [signals_before, header] = rdsamp(file_name);
        ann = rdann(file_name, 'fqrs'); % ��ȡע���ļ� 
        
        signals_before = signals_before(1:n_points_to_plot, :);
        ann = ann(ann <= n_points_to_plot);
        
        % ��ʼ���������ݴ洢
        data_0 = []; % 0������ݣ�������qrs
        data_1 = []; % 1������ݣ�����qrs

        % ��ʼ��ͨ���������ؾ���
        sample_entropy_channel = zeros(1, n_channels);
        for channel = 1:n_channels
                signal_before = signals_before(:, channel);
                [signals, fetal_QRSAnn_est, QT_Interval] = physionet2013(signal_before);
                signal = DenoiseSig(signals);
                signal = (signal - min(signal)) / (max(signal) - min(signal));
                signal_before = DenoiseSig(signal_before);
                signal_before = (signal_before - min(signal_before)) / (max(signal_before) - min(signal_before));
                
                % ��ʼ�������ص�����
                sample_entropy_array = zeros(1, 3);
        
                % �����ȡ��������Ӳ���
                for seg = 1:10
                    start_index = randi(n_points_to_plot - segment_length_2000 + 1);
                    end_index = start_index + segment_length_2000 - 1;

                    % ��ȡ��ǰ����
                    signal_2000 = signal(start_index:end_index, 1);
              
                    % ����������
                    m = 2; % Ƕ��ά��
                    std_percentage = 520; % �ٷֱȷ��еı�׼��ٷֱ�
                    r = std(signal_2000) * (std_percentage / 100); % ƥ����
                    sample_entropy = sample_entropy2(signal_2000, m, r);
                    
                    % ��������ֵ���浽������
                    sample_entropy_array(seg) = sample_entropy;
                end
        
                % ���ֵ�����浽ͨ���������ؾ�����
                sample_entropy_channel(channel) = mean(sample_entropy_array);
                if sample_entropy_channel(channel) > 0.55/100
                     for ii = 1:length(ann)
                        ann_point = ann(ii);
                        segment_length = 60;%40��-24��-15
                        % �������ann��ǵ������
                        for j = -34:2:-25
                            start_idx = ann_point + j;
                            end_idx = start_idx + segment_length - 1;
                
                            if start_idx > 0 && end_idx <= length(signal_before)
                                segment = signal_before(start_idx:end_idx);
                                data_1 = [data_1; segment'];
                            end
                        end
                
                        % ��������ann��ǵ������
                        for j = 1:2:10
                            start_idx = ann_point + 30 + (j - 1) * segment_length;
                            end_idx = ann_point + 30 + segment_length + (j - 1) * segment_length - 1;
                
                            if start_idx > 0 && end_idx < length(signal_before)
                                segment = signal_before(start_idx:end_idx);
                
                                % ����Ƿ��������ann��ǵ�
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
% plot(fqrs_indices(fqrs_indices <= n_points_to_plot), signal_before(fqrs_indices(fqrs_indices <= n_points_to_plot)), 'ro'); % ���� fqrs ��λ��
% title('����ǰ���ź� - ͨ��1');
% xlabel('ʱ��');
% ylabel('����');
% 
% % Subplot for signal
% subplot(2, 1, 2);
% plot(signal(1:n_points_to_plot));
% hold on;
% plot(fqrs_indices(fqrs_indices <= n_points_to_plot), signal(fqrs_indices(fqrs_indices <= n_points_to_plot)), 'ro'); % ���� fqrs ��λ��
% title('�������ź� - ͨ��1');
% xlabel('ʱ��');
% ylabel('����');
% % Adjusting figure properties for better visualization
% set(gcf, 'Position', [100, 100, 800, 600]); % Adjust figure size and position


% % ���� data_1 �� data_0 �Ѿ�������
% 
% % ѡ���������ݽ��л���
% data_1_samples = data_1(1:2, :);
% data_0_samples = data_0(1:2, :);
% 
% % ���� x �����꣨�������ݵ��ʱ����ȷֲ���
% x_axis = 1:50;
% 
% % ���� data_1 �е���������
% figure;
% subplot(2, 1, 1);
% plot(x_axis, data_1_samples(1, :), '-o', 'DisplayName', 'Data 1 - Row 1');
% hold on;
% plot(x_axis, data_1_samples(2, :), '-s', 'DisplayName', 'Data 1 - Row 2');
% title('ʾ��ͼ - data_1');
% legend;
% hold off;
% 
% % ���� data_0 �е���������
% subplot(2, 1, 2);
% plot(x_axis, data_0_samples(1, :), '-o', 'DisplayName', 'Data 0 - Row 1');
% hold on;
% plot(x_axis, data_0_samples(2, :), '-s', 'DisplayName', 'Data 0 - Row 2');
% title('ʾ��ͼ - data_0');
% legend;
% hold off;


%###
        % ���崰�ڴ�С�ͳ�ʼ������С����������Ϊ��λ��       
%         window_size = 100;%150;
%         step_size_no_qrs = 80;%50;
%         step_size_qrs = 12;%30;
%        
%         j = 1; % ��ʼ����ʼλ��
%         while j <= length(signals)-window_size+1
%             window_data = signals(j:j+window_size-1, :);
%             ann_window = ann(ann >= j & ann < j+window_size);
%     
%             if isempty(ann_window)
%                 data_0 = [data_0; window_data(:)'];
%                 j = j + step_size_no_qrs; % û��QRS��Ⱥ������ƶ�40����
%             else
%                 data_1 = [data_1; window_data(:)'];
%                 j = j + step_size_qrs; % ��QRS��Ⱥ������ƶ�10����
%             end
%         end
 %###   



%         % ��ÿһ�н��з�ת
%         flipped_data_0 = 1 - data_0;       
%         % ����ת���������ԭʼ����ƴ�ӣ���ʵ������������
%         augmented_data_0 = [data_0; flipped_data_0];
%         % ���ˣ�augmented_data_0 �����˶�ԭʼ���ݽ���ÿһ�����·�ת����1Ϊ���ߣ������ǿ����

        n_train = size(data_0,1);%��������,data_0��һ���������и�ɵĹ���Ϊ0�����У�100��
        for m = 1: n_train
            curPulse = data_0(m,:);
            curPulse(isnan(curPulse(1,:))==1)=[0];
            curPulse(isinf(curPulse(1,:))==1)=[1];
%             plot(curPulse);%hold on;
            Y0_1 = curPulse; %ʱ��1*80
            Y0_1 = reshape(Y0_1, segment_length, 1)';%2*40
            Y0_2 = abs(fft(curPulse));%Ƶ���źţ�1*80
            Y0_2 = Y0_2(1:segment_length);%1*40
            Y0_3 = Eigenvalue(curPulse);%����ֵ�źţ�1*40
            Y0_4 = cwt(curPulse); %plot(Y1);hold on;       %
            %size(Y0_4)
            Y0_4 = imresize(Y0_4,[24,segment_length],'bilinear');   
            Y0_4 = abs(Y0_4);
            Y0_4 = Y0_4/max(max(Y0_4))*2-1;%ʱƵ�� 34*40     mesh(Y1);
            Y0 = [Y0_1;Y0_2;Y0_3;Y0_4];

            if any(isnan(Y0(:)))
                disp('Y0 contains NaN values. Skipping...');
            else
                [Y_size1, Y_size2] = size(Y0);   
                fi = ['D:\��ʿ��\ѧϰ\���ķ���\�Ƿ������Ⱥ������\datanew\0_', num2str(i), '.txt'];
                fip = fopen(fi, 'a'); % �򿪻򴴽�Ҫд������ļ���׷�����ݵ��ļ�ĩβ��
                for ii = 1:Y_size1  
                    for j = 1:Y_size2  
                        fprintf(fip, '%g\t', Y0(ii, j));  % �����Ʊ���Ͱ���С������
                    end
                    fprintf(fip, '\r\n'); % �س� ����  
                end   
                fclose(fip);
            end
        end
    
%         % ��ÿһ�н��з�ת
%         flipped_data_1 = 1 - data_1;       
%         % ����ת���������ԭʼ����ƴ�ӣ���ʵ������������
%         augmented_data_1 = [data_1; flipped_data_1];
%         % ���ˣ�augmented_data_0 �����˶�ԭʼ���ݽ���ÿһ�����·�ת����1Ϊ���ߣ������ǿ����

         n_train = size(data_1,1); % ��������,data_1��һ���������и�ɵĹ���Ϊ1������
         for n = 1: n_train
            curPulse = data_1(n,:);
            curPulse(isnan(curPulse(1,:))==1)=[0];
            curPulse(isinf(curPulse(1,:))==1)=[1];
            % plot(curPulse);%hold on;
            Y1_1 = curPulse; %ʱ��1*40
            Y1_1 = reshape(Y1_1, segment_length, 1)';%2*40
            Y1_2 = abs(fft(curPulse));%Ƶ���źţ�1*40
            Y1_2 = Y1_2(1:segment_length);%1*40
            Y1_3 = Eigenvalue(curPulse);%����ֵ�źţ�1*40
            Y1_4 = cwt(curPulse); %plot(Y1);hold on;       %
            %size(Y1_4)
            Y1_4 = imresize(Y1_4,[24,segment_length],'bilinear');   
            Y1_4 = abs(Y1_4);
            Y1_4 = Y1_4/max(max(Y1_4))*2-1;%ʱƵ�� 34*40     mesh(Y1);
            Y1 = [Y1_1;Y1_2;Y1_3;Y1_4];
            
            [Y_size1, Y_size2] = size(Y1);   
            fi = ['D:\��ʿ��\ѧϰ\���ķ���\�Ƿ������Ⱥ������\datanew\1_', num2str(i), '.txt'];
            fip = fopen(fi, 'a'); % �򿪻򴴽�Ҫд������ļ���׷�����ݵ��ļ�ĩβ��
            
            % ���Y1�Ƿ����NaNֵ
            if any(isnan(Y1(:)))
                disp('Y1 contains NaN values. Skipping...');
            else
                for ii = 1:Y_size1  
                    for j = 1:Y_size2  
                        fprintf(fip, '%g\t', Y1(ii, j));  % �����Ʊ���Ͱ���С������
                    end
                    fprintf(fip, '\r\n'); % �س� ����  
                end   
            end
            fclose(fip);
         end
    else
        disp(['�ļ� ' file_name ' �����ڣ���������']);
    end 
end
fclose('all');
