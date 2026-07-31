clear; clc; close all;

%% ========== 设置路径 ==========
% 图片基础文件夹路径
base_folder = "D:\wurenji\outputnd";

% 坐标日志文件路径
log_file = "D:\gas\beamform_results_nd_cs\beamform_results_log.txt";

% 标注结果保存文件夹
output_folder = "D:\gas\marked_images_nd_cs_5_to_80";

%% ========== 距离与绘图参数 ==========
% 距离5到80，每隔5一个距离，共16层
distances = 5:5:80;

% 16种颜色，依次对应5、10、15……80
% 每一行是一个RGB颜色，RGB取值范围为0~1
colors = [
    1.00, 0.00, 0.00;   % 5  cm：红
    1.00, 0.30, 0.00;   % 10 cm：红橙
    1.00, 0.55, 0.00;   % 15 cm：橙
    1.00, 0.80, 0.00;   % 20 cm：金黄
    0.85, 0.90, 0.00;   % 25 cm：黄绿
    0.45, 0.80, 0.10;   % 30 cm：浅绿
    0.00, 0.65, 0.25;   % 35 cm：绿
    0.00, 0.70, 0.65;   % 40 cm：青绿
    0.00, 0.55, 0.90;   % 45 cm：天蓝
    0.00, 0.20, 1.00;   % 50 cm：蓝
    0.25, 0.00, 0.85;   % 55 cm：靛蓝
    0.45, 0.00, 0.75;   % 60 cm：深紫
    0.65, 0.00, 0.65;   % 65 cm：紫
    0.85, 0.00, 0.55;   % 70 cm：紫红
    0.35, 0.35, 0.35;   % 75 cm：深灰
    0.00, 0.00, 0.00    % 80 cm：黑
];

% 所有辅助点大小
marker_sizes = 14 * ones(size(distances));

% 圆周采样角度
theta = linspace(0, 2*pi, 720);

%% ========== 读取坐标日志 ==========
fprintf('读取坐标日志: %s\n', log_file);

fid = fopen(log_file, 'r');
if fid == -1
    error('无法打开日志文件: %s', log_file);
end

% 读取表头
header = fgetl(fid); %#ok<NASGU>

% 读取数据
data = textscan( ...
    fid, ...
    '%d %s %s %f %f %s %s', ...
    'Delimiter', '\t', ...
    'MultipleDelimsAsOne', false ...
);
fclose(fid);

% 解析数据
seq_nums    = data{1};
video_names = data{2}; %#ok<NASGU>
image_names = data{3};
x_coords    = data{4};
y_coords    = data{5};
result_files = data{6}; %#ok<NASGU>
statuses    = data{7};

record_count = length(seq_nums);
fprintf('共读取到 %d 条坐标记录\n\n', record_count);

%% ========== 检查数据长度 ==========
column_lengths = [
    length(seq_nums), ...
    length(image_names), ...
    length(x_coords), ...
    length(y_coords), ...
    length(statuses)
];

if any(column_lengths ~= record_count)
    error('日志各列数据长度不一致，请检查日志文件格式。');
end

%% ========== 显示汇总信息 ==========
fprintf('========== 坐标汇总 ==========\n');
fprintf('%-6s %-30s %8s %8s %s\n', ...
    '序号', '图片名称', 'X坐标', 'Y坐标', '状态');
fprintf('%-6s %-30s %8s %8s %s\n', ...
    '----', '------------------------------', '------', '------', '----');

for i = 1:record_count
    fprintf('%-6d %-30s %8.2f %8.2f %s\n', ...
        seq_nums(i), image_names{i}, ...
        x_coords(i), y_coords(i), statuses{i});
end

%% ========== 创建输出文件夹 ==========
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
    fprintf('创建输出文件夹: %s\n', output_folder);
end

%% ========== 逐张图片绘制坐标点 ==========
fprintf('\n========================================\n');
fprintf('开始逐张绘制图片和坐标点\n');
fprintf(' - 红色大实心点：原始中心坐标\n');
fprintf(' - 彩色辅助点：距离5、10、15……80的八方向点\n');
fprintf(' - 彩色虚线圆：各个距离对应的圆周\n');
fprintf(' - 最外层黑色点和黑色虚线圆：距离80\n');
fprintf('========================================\n\n');

success_count = 0;

% 缓存已经找到的图片文件夹，避免重复搜索
folder_cache = containers.Map( ...
    'KeyType', 'char', ...
    'ValueType', 'char' ...
);

for i = 1:record_count
    %% ----- 从图片名称提取时间戳 -----
    img_base_name = image_names{i};

    % 格式示例：HM20260702_111044
    if length(img_base_name) >= 17
        timestamp_part = img_base_name(1:17);
    else
        fprintf('[%d/%d] 图片名称格式异常: %s，跳过\n', ...
            i, record_count, img_base_name);
        continue;
    end

    %% ----- 查找对应的图片文件夹 -----
    if isKey(folder_cache, timestamp_part)
        image_folder = folder_cache(timestamp_part);
    else
        expected_folder_name = [timestamp_part, '.ld_frames'];

        % 第一候选位置
        image_folder = fullfile(base_folder, expected_folder_name);

        % 第二候选位置
        alt_base = 'D:\gas\output';
        if ~exist(image_folder, 'dir')
            image_folder = fullfile(alt_base, expected_folder_name);
        end

        % 精确目录仍不存在时，搜索包含时间戳的目录
        if ~exist(image_folder, 'dir')
            found = false;

            if exist(base_folder, 'dir')
                dir_list = dir(base_folder);

                for k = 1:length(dir_list)
                    if dir_list(k).isdir && ...
                            ~strcmp(dir_list(k).name, '.') && ...
                            ~strcmp(dir_list(k).name, '..') && ...
                            contains(dir_list(k).name, timestamp_part)

                        image_folder = fullfile(base_folder, dir_list(k).name);
                        found = true;
                        break;
                    end
                end
            end

            if ~found && exist(alt_base, 'dir')
                dir_list2 = dir(alt_base);

                for k = 1:length(dir_list2)
                    if dir_list2(k).isdir && ...
                            ~strcmp(dir_list2(k).name, '.') && ...
                            ~strcmp(dir_list2(k).name, '..') && ...
                            contains(dir_list2(k).name, timestamp_part)

                        image_folder = fullfile(alt_base, dir_list2(k).name);
                        found = true;
                        break;
                    end
                end
            end

            if ~found
                fprintf('[%d/%d] 找不到时间戳 %s 对应的图片文件夹，跳过\n', ...
                    i, record_count, timestamp_part);
                continue;
            end
        end

        % 缓存找到的目录
        folder_cache(timestamp_part) = image_folder;
        fprintf('找到图片文件夹: %s\n', image_folder);
    end

    %% ----- 构建并读取图片 -----
    img_name = [image_names{i}, '.jpg'];
    img_path = fullfile(image_folder, img_name);

    if ~exist(img_path, 'file')
        fprintf('[%d/%d] 图片不存在: %s\n', ...
            i, record_count, img_path);
        continue;
    end

    try
        img = imread(img_path);
    catch ME
        fprintf('[%d/%d] 无法读取图片: %s\n原因: %s\n', ...
            i, record_count, img_path, ME.message);
        continue;
    end

    %% ----- 创建隐藏图窗 -----
    fig = figure( ...
        'Name', sprintf('图片 %d/%d: %s', i, record_count, img_name), ...
        'Position', [100, 100, 1200, 850], ...
        'Visible', 'off', ...
        'Color', 'white' ...
    );

    ax = axes(fig);
    imshow(img, 'Parent', ax);
    hold(ax, 'on');

    cx = x_coords(i);
    cy = y_coords(i);

    % 原始中心点
    h_original = plot( ...
        ax, cx, cy, ...
        'r.', ...
        'MarkerSize', 26 ...
    );

    %% ----- 绘制5到80的全部辅助点 -----
    legend_handles = gobjects(1, length(distances) + 1);
    legend_labels = cell(1, length(distances) + 1);

    legend_handles(1) = h_original;
    legend_labels{1} = '原始坐标';

    for d = 1:length(distances)
        offset = distances(d);
        current_color = colors(d, :);

        % 八个方向：
        % 上、下、左、右、左上、左下、右上、右下
        diagonal_offset = offset / sqrt(2);

        points_x = [
            cx, ...
            cx, ...
            cx - offset, ...
            cx + offset, ...
            cx - diagonal_offset, ...
            cx - diagonal_offset, ...
            cx + diagonal_offset, ...
            cx + diagonal_offset
        ];

        points_y = [
            cy - offset, ...
            cy + offset, ...
            cy, ...
            cy, ...
            cy - diagonal_offset, ...
            cy + diagonal_offset, ...
            cy - diagonal_offset, ...
            cy + diagonal_offset
        ];

        % 绘制八方向点
        plot( ...
            ax, points_x, points_y, ...
            '.', ...
            'Color', current_color, ...
            'MarkerSize', marker_sizes(d) ...
        );

        % 绘制当前距离圆
        circle_x = cx + offset * cos(theta);
        circle_y = cy + offset * sin(theta);

        plot( ...
            ax, circle_x, circle_y, ...
            '--', ...
            'Color', current_color, ...
            'LineWidth', 0.9 ...
        );

        % 创建图例虚拟句柄
        legend_handles(d + 1) = plot( ...
            ax, nan, nan, ...
            '.', ...
            'Color', current_color, ...
            'MarkerSize', 16 ...
        );

        legend_labels{d + 1} = sprintf('距离%d', offset);
    end

    %% ----- 标注中心点信息 -----
    text_offset = 28;

    text( ...
        ax, ...
        cx + text_offset, ...
        cy - text_offset, ...
        sprintf('#%d (%.1f, %.1f)', seq_nums(i), cx, cy), ...
        'Color', 'red', ...
        'FontSize', 11, ...
        'FontWeight', 'bold', ...
        'BackgroundColor', 'white', ...
        'EdgeColor', 'red', ...
        'Margin', 3 ...
    );

    %% ----- 标题和图例 -----
    title_str = sprintf( ...
        '[%d/%d] %s | 坐标: (%.1f, %.1f) | 状态: %s | 距离5-80', ...
        i, record_count, img_name, cx, cy, statuses{i} ...
    );

    title( ...
        ax, title_str, ...
        'FontSize', 12, ...
        'Interpreter', 'none' ...
    );

    % 16个距离加中心点，共17项，使用4列减少高度
    legend( ...
        ax, ...
        legend_handles, ...
        legend_labels, ...
        'Location', 'northeastoutside', ...
        'FontSize', 8, ...
        'NumColumns', 4 ...
    );

    hold(ax, 'off');

    %% ----- 保存标注图片 -----
    output_img_name = [image_names{i}, '_marked_5_to_80.jpg'];
    output_path = fullfile(output_folder, output_img_name);

    try
        % 优先使用exportgraphics，可更好地保存图例和文字
        exportgraphics(fig, output_path, 'Resolution', 150);
    catch
        % 兼容较旧版本MATLAB
        saveas(fig, output_path, 'jpg');
    end

    close(fig);

    success_count = success_count + 1;

    fprintf('[%d/%d] 图片处理完成: %s\n', ...
        i, record_count, output_path);

    if mod(i, 10) == 0 || i == record_count
        fprintf('当前进度：已扫描 %d/%d 条，成功处理 %d 张\n', ...
            i, record_count, success_count);
    end
end

%% ========== 最终统计 ==========
fprintf('\n========================================\n');
fprintf('所有图片处理完成！\n');
fprintf('成功处理: %d/%d 张图片\n', success_count, record_count);
fprintf('保存路径: %s\n', output_folder);
fprintf('已绘制距离: ');
fprintf('%d ', distances);
fprintf('\n========================================\n');
