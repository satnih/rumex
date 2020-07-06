% read image

xmlfile = '/u/21/hiremas1/unix/postdoc/rumex/data/aerial/original/WENR_ortho_Rumex_10m_2_sw.xml';
image_file = '/u/21/hiremas1/unix/postdoc/rumex/data/aerial/original/WENR_ortho_Rumex_10m_2_sw.png';
im = imread(image_file);
bbox = get_bbox_from_xml(xmlfile);
nboxes = length(bbox);
figure
imshow(im)
hold on
patches = cell(nboxes, 1);
for i = 1:nboxes
    curbb = bbox(i, 1:4);
    xmin = curbb(1);
    ymin = curbb(2);
    xmax = curbb(3);
    ymax = curbb(4);
    width = xmax - xmin;
    height = ymax - ymin;    
    patches{i} = im(ymin:ymax, xmin:xmax, :);
    rectangle('Position', [xmin, ymin, width, height], 'EdgeColor', 'r', 'LineWidth', 2)
    if (xmin < x_crop && xmax && x_crop && ymin < y_crop && ymax < y_crop)
        rectangle('Position', [xmin, ymin, width, height], 'EdgeColor', 'r', 'LineWidth', 2)
    end
end
hold off


function bboxs = get_bbox_from_xml(fname)
    xml = myxml2struct(fname);
    rumex_struct = xml.annotation.object;
    nobjects = size(rumex_struct, 2);
    bboxs = zeros(nobjects, 5);
    for i = 1:nobjects
        difficult_flag = str2double(rumex_struct{i}.difficult.Text);
        xmin = str2double(rumex_struct{i}.bndbox.xmin.Text);
        ymin = str2double(rumex_struct{i}.bndbox.ymin.Text);
        xmax = str2double(rumex_struct{i}.bndbox.xmax.Text);
        ymax = str2double(rumex_struct{i}.bndbox.ymax.Text);
        bboxs(i, :) = [xmin, ymin, xmax, ymax, difficult_flag];
    end
end