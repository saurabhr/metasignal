function plot_corrTables(r, p, variable_names, subplotTitles)

% Open a new figure
figure('Color','w', 'DefaultAxesFontSize',12);

% Loop over all measures
for dset=1:length(r)
    % Create relevant subplot
    ax = subplot(1,3,dset);
    imagesc(r{dset});
    hold on
    
    % Add p values
    for meas1=1:size(r{dset},1)
        for meas2=1:size(r{dset},1)
            if p{dset}(meas1,meas2) < .05
                text(meas2,meas1, '*', 'FontSize', 14)
            end
        end
    end
    
    % Add details to the plot
    caxis([-1, 1]);
    set(ax,'XTick', 1:size(r{dset},1));
    set(gca,'XTickLabel',variable_names);
    set(ax,'YTick', 1:size(r{dset},1));
    set(gca,'YTickLabel',variable_names);
    xtickangle(90);
    title(subplotTitles{dset}, 'FontSize', 18);
end
ylabel(colorbar,'r-value','FontSize',16);