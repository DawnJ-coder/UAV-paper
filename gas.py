消声室参考组代码	消声室参考组中文	场景数	样本配对数	RAW真泄漏马氏距离	PLANE真泄漏马氏距离	RAW假泄漏马氏距离	PLANE假泄漏马氏距离	RAW真泄漏频谱相似度	PLANE真泄漏频谱相似度	RAW假泄漏频谱相似度	PLANE假泄漏频谱相似度	RAW马氏距离真假分离_FALSE减TRUE	PLANE马氏距离真假分离_FALSE减TRUE	马氏距离分离改善_PLANE减RAW	RAW频谱相似度真假分离_TRUE减FALSE	PLANE频谱相似度真假分离_TRUE减FALSE	频谱相似度分离改善_PLANE减RAW	明显改善样本数	部分改善样本数	明显变差样本数	部分变差样本数	变化很小或矛盾样本数
0.1mm	0.1mm泄漏参考	3	44	4.836270732	5.062204446	7.280068065	10.29006314	0.910355126	0.882431854	0.832733177	0.822132303	2.443797333	5.227858694	2.784061361	0.077621949	0.060299551	-0.017322398	11	6	2	9	16
0.5mm	0.5mm泄漏参考	3	44	4.667454214	6.174430341	6.107450597	11.15807411	0.924353954	0.889548395	0.813138277	0.791911665	1.439996383	4.98364377	3.543647386	0.111215677	0.09763673	-0.013578947	13	7	7	10	7
ALL_LEAKS	0.1mm+0.5mm联合泄漏参考	3	44	4.089655361	4.312880107	6.186465583	9.311781008	0.921252332	0.890987182	0.836692047	0.823641289	2.096810222	4.998900902	2.902090679	0.084560285	0.067345893	-0.017214392	10	9	3	11	11


消声室参考组代码	消声室参考组中文	场景数	样本配对数	RAW真泄漏马氏距离	PLANE真泄漏马氏距离	RAW假泄漏马氏距离	PLANE假泄漏马氏距离	RAW真泄漏频谱相似度	PLANE真泄漏频谱相似度	RAW假泄漏频谱相似度	PLANE假泄漏频谱相似度	RAW马氏距离真假分离_FALSE减TRUE	PLANE马氏距离真假分离_FALSE减TRUE	马氏距离分离改善_PLANE减RAW	RAW频谱相似度真假分离_TRUE减FALSE	PLANE频谱相似度真假分离_TRUE减FALSE	频谱相似度分离改善_PLANE减RAW	明显改善样本数	部分改善样本数	明显变差样本数	部分变差样本数	变化很小或矛盾样本数
0.1mm	0.1mm泄漏参考	3	44	3.987798518	3.999731127	3.970756414	5.887580543	0.872044538	0.878700058	0.811665601	0.74563352	-0.017042104	1.887849416	1.904891521	0.060378936	0.133066538	0.072687601	20	3	0	7	14
0.5mm	0.5mm泄漏参考	3	44	2.797400001	2.813260562	5.056043018	6.751581815	0.886481302	0.875662629	0.792054927	0.710201018	2.258643017	3.938321253	1.679678236	0.094426375	0.165461611	0.071035236	20	2	0	7	15
ALL_LEAKS	0.1mm+0.5mm联合泄漏参考	3	44	2.593335476	2.538258124	3.683623225	5.035831088	0.881005051	0.883867232	0.813920142	0.741880994	1.090287749	2.497572963	1.407285214	0.067084909	0.141986237	0.074901328	18	5	0	7	14

clear; clc; close all;

%% ========== 设置路径 ==========
% 图片基础文件夹路径
base_folder = "D:\wurenji\outputnd";
% 坐标日志文件路径
log_file = "D:\gas\beamform_results_nd_cs\beamform_results_log.txt";

%% ========== 读取坐标日志 ==========
fprintf('读取坐标日志: %s\n', log_file);
% 打开日志文件
fid = fopen(log_file, 'r');
if fid == -1
    error('无法打开日志文件: %s', log_file);
end

% 读取表头（第一行）
header = fgetl(fid);
% 读取数据
data = textscan(fid, '%d %s %s %f %f %s %s', 'Delimiter', '\t');
fclose(fid);

% 解析数据
seq_nums = data{1};
video_names = data{2};
image_names = data{3};
x_coords = data{4};
y_coords = data{5};
result_files = data{6};
statuses = data{7};

fprintf('共读取到 %d 条坐标记录\n\n', length(seq_nums));

%% ========== 显示汇总信息 ==========
fprintf('========== 坐标汇总 ==========\n');
fprintf('%-6s %-30s %8s %8s %s\n', '序号', '图片名称', 'X坐标', 'Y坐标', '状态');
fprintf('%-6s %-30s %8s %8s %s\n', '----', '------------------------------', '------', '------', '----');

for i = 1:length(seq_nums)
    fprintf('%-6d %-30s %8.2f %8.2f %s\n', ...
        seq_nums(i), image_names{i}, x_coords(i), y_coords(i), statuses{i});
end

%% ========== 逐张图片显示坐标点 ==========
fprintf('\n========================================\n');
fprintf('开始逐张显示图片和坐标点...\n');
fprintf('操作说明：\n');
fprintf(' - 红色实心点：原始坐标点\n');
fprintf(' - 红橙黄绿蓝靛紫黑八色点：距离5/10/15/20/25/30/35/40的辅助点\n');
fprintf('========================================\n\n');

% 定义距离数组和对应的颜色（使用RGB值）
distances = [5, 10, 15, 20, 25, 30, 35, 40];
% 红橙黄绿蓝靛紫黑 对应的RGB值
colors = {
    [1, 0, 0],        % 红色 - 距离5
    [1, 0.5, 0],      % 橙色 - 距离10
    [1, 1, 0],        % 黄色 - 距离15
    [0, 1, 0],        % 绿色 - 距离20
    [0, 0, 1],        % 蓝色 - 距离25
    [0.29, 0, 0.51],  % 靛色 - 距离30
    [0.5, 0, 0.5],    % 紫色 - 距离35
    [0, 0, 0]         % 黑色 - 距离40
};

marker_sizes = [14, 14, 14, 14, 14, 14, 14, 14]; % 调整点的大小使其更明显

% 创建保存结果的文件夹
output_folder = 'D:\gas\marked_images_nd_cs';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
    fprintf('创建输出文件夹: %s\n', output_folder);
end

% 统计成功处理的图片数量
success_count = 0;

% 用于缓存已找到的文件夹路径，避免重复查找
folder_cache = containers.Map();

for i = 1:length(seq_nums)
    % 从图片名称中提取时间戳部分（如：HM20260702_111044）
    img_base_name = image_names{i};
    
    % 提取时间戳部分（前17个字符，格式：HMYYYYMMDD_HHMMSS）
    if length(img_base_name) >= 17
        timestamp_part = img_base_name(1:17);
    else
        fprintf('[%d/%d] 图片名称格式异常: %s，跳过\n', i, length(seq_nums), img_base_name);
        continue;
    end
    
    % 根据时间戳查找对应的图片文件夹
    if isKey(folder_cache, timestamp_part)
        image_folder = folder_cache(timestamp_part);
    else
        % 构建预期的文件夹名
        expected_folder_name = [timestamp_part, '.ld_frames'];
        image_folder = fullfile(base_folder, expected_folder_name);
        
        % 如果 base_folder 下找不到，尝试在 D:\gas\output 下查找
        if ~exist(image_folder, 'dir')
            alt_base = 'D:\gas\output';
            image_folder = fullfile(alt_base, expected_folder_name);
        end

        
        % 如果还找不到，尝试搜索匹配的文件夹
        if ~exist(image_folder, 'dir')
            % 在 base_folder 下搜索包含时间戳的文件夹
            if exist(base_folder, 'dir')
                dir_list = dir(base_folder);
                found = false;
                for k = 1:length(dir_list)
                    if dir_list(k).isdir && contains(dir_list(k).name, timestamp_part)
                        image_folder = fullfile(base_folder, dir_list(k).name);
                        found = true;
                        break;
                    end
                end
                
                % 在 D:\gas\output 下也搜索
                if ~found && exist('D:\gas\output', 'dir')
                    dir_list2 = dir('D:\gas\output');
                    for k = 1:length(dir_list2)
                        if dir_list2(k).isdir && contains(dir_list2(k).name, timestamp_part)
                            image_folder = fullfile('D:\gas\output', dir_list2(k).name);
                            found = true;
                            break;
                        end
                    end
                end
                
                if ~found
                    fprintf('[%d/%d] 找不到时间戳 %s 对应的图片文件夹，跳过\n', ...
                        i, length(seq_nums), timestamp_part);
                    continue;
                end
            else
                fprintf('[%d/%d] 基础文件夹不存在: %s，跳过\n', i, length(seq_nums), base_folder);
                continue;
            end
        end
        
        % 缓存找到的文件夹路径
        folder_cache(timestamp_part) = image_folder;
        fprintf('找到图片文件夹: %s\n', image_folder);
    end
    
    % 构建图片文件名
    img_name = [image_names{i}, '.jpg'];
    img_path = fullfile(image_folder, img_name);
    
    % 检查图片是否存在
    if ~exist(img_path, 'file')
        fprintf('[%d/%d] 图片不存在: %s\n', i, length(seq_nums), img_path);
        continue;
    end
    
    % 读取图片
    try
        img = imread(img_path);
    catch
        fprintf('[%d/%d] 无法读取图片: %s\n', i, length(seq_nums), img_path);
        continue;
    end
    
    % 创建图形窗口（隐藏窗口以提高速度）
    fig = figure('Name', sprintf('图片 %d/%d: %s', i, length(seq_nums), img_name), ...
                 'Position', [100, 100, 900, 700], 'Visible', 'off');
    
    % 显示图片
    imshow(img);
    hold on;
    
    % 获取当前坐标
    cx = x_coords(i);
    cy = y_coords(i);
    
    % 标注原始坐标点（红色实心大圆点）
    plot(cx, cy, 'r.', 'MarkerSize', 24);
    
    % 为每个距离绘制辅助点（八个方向）
    legend_handles = [];
    legend_labels = {};
    
    for d = 1:length(distances)
        offset = distances(d);
        color = colors{d};
        
        % 八个方向的坐标计算
        % 上下左右 + 四个对角线方向
        points_x = [cx, cx, cx-offset, cx+offset, ...
                   cx-offset/sqrt(2), cx-offset/sqrt(2), cx+offset/sqrt(2), cx+offset/sqrt(2)];
        points_y = [cy-offset, cy+offset, cy, cy, ...
                   cy-offset/sqrt(2), cy+offset/sqrt(2), cy-offset/sqrt(2), cy+offset/sqrt(2)];
        
        % 绘制八个方向的点
        plot(points_x, points_y, '.', 'Color', color, 'MarkerSize', marker_sizes(d));
        
        % 绘制圆形轮廓（帮助看清距离）
        theta = 0:0.01:2*pi;
        circle_x = cx + offset * cos(theta);
        circle_y = cy + offset * sin(theta);
        plot(circle_x, circle_y, '--', 'Color', color, 'LineWidth', 1);
        
        % 为图例创建句柄
        legend_handles(d) = plot(nan, nan, '.', 'Color', color, 'MarkerSize', 16);
        legend_labels{d} = sprintf('距离%d', distances(d));
    end
    
    % 在坐标点旁边显示序号和坐标值
    text_offset = 28;
    text(cx + text_offset, cy - text_offset, ...
         sprintf('#%d (%.1f, %.1f)', seq_nums(i), cx, cy), ...
         'Color', 'red', 'FontSize', 11, 'FontWeight', 'bold', ...
         'BackgroundColor', 'white', 'EdgeColor', 'red');
    
    % 添加标题
    title_str = sprintf('[%d/%d] %s | 坐标: (%.1f, %.1f) | 状态: %s', ...
                        i, length(seq_nums), img_name, ...
                        cx, cy, statuses{i});
    title(title_str, 'FontSize', 12, 'Interpreter', 'none');
    
    % 添加原始坐标点到图例
    h_original = plot(nan, nan, 'r.', 'MarkerSize', 24);
    legend_handles = [h_original, legend_handles];
    legend_labels = [{'原始坐标'}, legend_labels];
    

    % 添加图例（分两列显示以避免过长）
    legend(legend_handles, legend_labels, ...
           'Location', 'northeast', 'FontSize', 8, 'NumColumns', 2);
    hold off;
    
    % 保存图片
    output_img_name = [image_names{i}, '_marked.jpg'];
    output_path = fullfile(output_folder, output_img_name);
    saveas(fig, output_path, 'jpg');
    
    % 关闭当前图片
    close(fig);
    
    % 统计成功数量
    success_count = success_count + 1;
    fprintf('[%d/%d] 图片处理完成: %s\n', i, length(seq_nums), img_path);
    
    % 每处理10张图片显示一次进度
    if mod(i, 10) == 0 || i == length(seq_nums)
        fprintf('已完成 %d/%d 张图片的处理\n', i, length(seq_nums));
    end
end

fprintf('\n所有图片处理完成！\n');
fprintf('成功处理: %d/%d 张图片\n', success_count, length(seq_nums));
fprintf('保存路径: %s\n', output_folder);


