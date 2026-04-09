function colors = good_colors_for_plotting(num_colors)

% Check input
if num_colors > 8
    
end

% Colors appropriate for all types of color blindness (as suggested in
% medium.com/@courtneyjordan/designing-for-all-users-why-you-should-care-about-color-blindness-beabd61943eb)
all_colors = {[86,32,70]/255, [150,32,56]/255, [212,49,56]/255, [240,102,60]/255, ...
    [246,150,105]/255, [253,203,153]/255, [241,239,142]/255, [250,247,204]/255};

switch num_colors
    case 1
        colors = all_colors{3};
    case 2
        colors = {all_colors{3}, all_colors{6}};
    case 3
        colors = {all_colors{2}, all_colors{4}, all_colors{6}};
    case 4
        colors = {all_colors{2}, all_colors{3}, all_colors{4}, all_colors{6}};
    case 5
        colors = {all_colors{1}, all_colors{2}, all_colors{3}, all_colors{5}, all_colors{7}};
    case 6
        colors = {all_colors{2}, all_colors{3}, all_colors{4}, all_colors{5}, all_colors{6}, all_colors{7}};
    case 7
        colors = {all_colors{1}, all_colors{2}, all_colors{3}, all_colors{4}, all_colors{5}, all_colors{6}, all_colors{7}};
    case 8
        colors = all_colors;
    otherwise
        fprintf(['Number of colors between 1 and 8 can be requested. Requested colors: ' num2str(num_colors) '\n'])
        return;
end
            
