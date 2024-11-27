%% Draw a virtual city
Map_story = zeros(100,100);
Building_story = readmatrix(['InputData\','Virtual City Building.xlsx'],'Sheet','ID');
Prob_exceed = readmatrix(['InputData\','Prob_Building.xlsx'],'Sheet','PFA_08');
rng(1e2); % 'default' 1e2 5e2 1e3 5e3
loc_bui = randperm(100*100,1000);
Map_story(loc_bui) = Building_story(:,5);

figure(2)
% plot prob on the buildings
values = Prob_exceed(:,4); % Extensive damage for safe or unsafe occupy
colormap('jet'); % jet parula
cmap = colormap;
% data to color
colorIndices = ceil(values*size(cmap,1));
mappedColors = cmap(colorIndices,:);

b = bar3(Map_story,0.85,'detached');
for k = 1:length(b)
    b(k).FaceColor = mappedColors(k,:);
    b(k).EdgeAlpha = 0.8;
    b(k).LineWidth = 0.6;
end
colorbar

xlim([-0.2 100.5]);ylim([-0.2 100.5]);zlim([0.01 12.1]);
xticks(0:20:100);
yticks(0:20:100);
zticks(0:6:12);
%Set label
set(gca,'xticklabel',{'0','1','2','3','4','5'},'Fontname','Times New Roman','FontSize',16);
set(gca,'yticklabel',{'0','1','2','3','4','5'},'Fontname','Times New Roman','FontSize',16);
%Modify the labels for each axis
xlabel('X Direction (km)','Fontname','Times New Roman','FontSize',18,'Rotation',18,'Position',[32 104 -3]);
ylabel('Y Direction (km)','Fontname','Times New Roman','FontSize',18,'Rotation',-32,'Position',[-2 56 -4]);
zlabel('Building Story','Fontname','Times New Roman','FontSize',16);
set(gca,'looseInset',[0 0 0.02 0.02]);
grid off
view(-45,75); % 3D view

%% 
% Map_prob = zeros(100,100);
% Map_prob(loc_bui) = Prob_exceed(:,2);
% figure(2)
% h = heatmap(Map_prob,'Colormap',parula);
% ax = gca;
% ax.XDisplayLabels = nan(size(ax.XDisplayData));
% ax.YDisplayLabels = nan(size(ax.YDisplayData));
% set(gca,'FontName','Times New Roman','FontSize',14)
