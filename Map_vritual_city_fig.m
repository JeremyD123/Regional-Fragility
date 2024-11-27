clear;
Fig = 1;
switch Fig
    case(1)
        %% Draw a virtual city
        Map_story = zeros(100,100);
        Building_story = readmatrix(['InputData\','Virtual City Building.xlsx'],'Sheet','ID');
        rng(1e2); % 'default' 1e2 5e2 1e3 5e3
        loc_bui = randperm(100*100,1000);
        Map_story(loc_bui) = Building_story(:,5);

        figure(3)
        b = bar3(Map_story,0.85,'detached');
        for k = 1:length(b)
            b(k).FaceColor = [0.5 0.5 0.5];
            b(k).EdgeAlpha = 0.8;
            b(k).LineWidth = 0.6;
        end

        xlim([-0.2 100.5]);ylim([-0.2 100.5]);zlim([0.01 12.1]);
        xticks(0:20:100);
        yticks(0:20:100);
        zticks(0:6:12);
        %Set label
        set(gca,'xticklabel',{'0','1','2','3','4','5'},'Fontname','Times New Roman','FontSize',16);
        set(gca,'yticklabel',{'0','1','2','3','4','5'},'Fontname','Times New Roman','FontSize',16);
        %Modify the labels for each axis
        xlabel('X Direction (km)','Fontname','Times New Roman','FontSize',18,'Rotation',28,'Position',[32 104 -3]);
        ylabel('Y Direction (km)','Fontname','Times New Roman','FontSize',18,'Rotation',-28,'Position',[-2 56 -4]);
        zlabel('Building Story','Fontname','Times New Roman','FontSize',16);
        set(gca,'looseInset',[0 0 0.02 0.02]);
        grid off
        view(-45,75); % 3D view
    case(2)
        data = readmatrix('InputData\Building inventory.xlsx','Sheet','ID');
        Const_period = data(:,2);
        Building_story = data(:,3);
        Area_floor = data(:,4);
        
        figure(3)
        h1 = histogram(Building_story(1:600,:),'LineWidth',1.0);
        h1.FaceColor = '#0072BD';
        hold on
        h2 = histogram(Building_story(601:end,:),'LineWidth',1.0);
        h2.FaceColor = '#77AC30';
        hold off
        legend('RC frames','RC frame-shear walls','FontName','Times New Roman','FontSize',16);
        xlim([0,12]);
        ylim([0,158]);
        xlabel('Number of storey','FontSize',20);
        ylabel('Number of buildings','FontSize',20);
        set(gca,'LooseInset',[0.04,0.06,0.02,0.02],'FontName','Times New Roman','FontSize',18,LineWidth=2.0)

        figure(4)
        h = histogram(Area_floor,30,'LineWidth',1.0);
        h.FaceColor = '#77AC30';
        xlim([0,2500]);
        xlabel('Floor area (m^2)','FontSize',20);
        ylabel('Number of buildings','FontSize',20);
        set(gca,'LooseInset',[0.04,0.06,0.02,0.02],'FontName','Times New Roman','FontSize',18,LineWidth=2.0)
    case(3)
        Dxy = readmatrix('InputData\Building inventory.xlsx','Sheet','MR');
        Xrange = [5.5,8];
        Yrange = [0,100];
        Xn = 5;Yn = 5;
        X = linspace(Xrange(1),Xrange(2),Xn+1);
        Y = linspace(Yrange(1),Yrange(2),Yn+1);
        Dx = Dxy(:,1); Dy = Dxy(:,2);
        H = zeros(Yn,Xn) ;
        for i = 1:length(Dxy)
            x = find(X > Dx(i));
            x = x(1)-1;
            y = find(Y > Dy(i));
            y = y(1)-1;
            H(y,x) = H(y,x)+1;
        end
        figure(2)
        b = bar3(H,0.70,'detached');
        xlim([0.5 5.5]);ylim([0.5 5.5]);
        xticks(0.5:1:4.5);
        yticks(1.5:1:5.5);
        %Set label
        set(gca,'xticklabel',{'5.5','6.0','6.5','7.0','7.5'});
        set(gca,'yticklabel',{'20','40','60','80','100'});
        %Modify the labels for each axis
        xlabel('Magnitude','Rotation',14,'Position',[1.1 9.5 7.4]);
        ylabel('Rupture distance (km)','Rotation',-23,'Position',[-2.5 6.6 16.7]);
        zlabel('Number');
        set(gca,'LooseInset',[0.04,0.06,0.02,0.02],'FontName','Times New Roman','FontSize',16,LineWidth=1.5)
    case(4)
        % Parameters
        ratio = 0.05;
        T1 = 0.8; % natural period of the considered structure
        load('InputData\GMRs\dt_peer.mat','dt')
        sPeriod = [0.01 0.02 0.022 0.025 0.029 0.03 0.032 0.035 0.036 0.04 ...
            0.042 0.044 0.045 0.046 0.048 0.05 0.055 0.06 0.065 0.067 0.07 ...
            0.075 0.08 0.085 0.09 0.095 0.1 0.11 0.12 0.13 0.133 0.14 0.15 ...
            0.16 0.17 0.18 0.19 0.2 0.22 0.24 0.25 0.26 0.28 0.29 0.3 0.32 ...
            0.34 0.35 0.36 0.38 0.4 0.42 0.44 0.45 0.46 0.48 0.5 0.55 0.6 ...
            0.65 0.667 0.7 0.75 0.8 0.85 0.9 0.95 1 1.1 1.2 1.3 1.4 1.5 1.6 ...
            1.7 1.8 1.9 2 2.2 2.4 2.5 2.6 2.8 3 3.2 3.4 3.5 3.6 3.8 4 4.2 ...
            4.4 4.6 4.8 5 5.5 6];
        indPer = [27 29 32 34 37 39 42 45 49 52 56 58 61 64	66 69 71 73 75 78 80 83	85 89 93 96];
        % parameters of the design response spectrum
        a_max = 0.72;
        Tg = 0.35;
        gamma = 0.9;
        eta1 = 0.02;
        eta2 = 1;
        %% Design response spectrum
        t1 = sPeriod(1:27);      %0.1:dt:0.1
        t2 = sPeriod(28:48);     %0.1+dt:dt:Tg
        t3 = sPeriod(49:75);     %Tg+dt:dt:5*Tg-dt
        t4 = sPeriod(76:end);    %5*Tg:dt:T(end);
        Sa_design = [0.45*a_max+(eta2-0.45)*a_max/0.1*t1,eta2*a_max*ones(1,length(t2)),...
            (Tg./t3).^gamma*eta2*a_max,(eta2*0.2^gamma-eta1*(t4-5*Tg))*a_max];
        % k1 = round(T1/dt);% point of the structure period

        %% real earthquake
        % Define ground motion input
        numGM = 100;
        Sa_database = zeros(numGM,length(sPeriod));

        for i = 1:numGM
            % GROUND MOTION INPUT
            fid = fopen(['InputData\GMRs\GM_',num2str(i),'.txt'], 'r');
            gmr = fscanf(fid,'%f');   % ground motion (g)
            gacc = gmr/9.81;
            fclose(fid);
            % Spectral solution
            for j = 1:length(sPeriod)
                [sa,sv,sd]=spectrasa(dt(i),gacc,sPeriod(j),ratio);
                Sa_database(i,j)  = sa;                       % Sa (g)
            end
        end

        %% Plot response spectra all
        figure(1)
        for i = 1:numGM
            L2 = plot(sPeriod,Sa_database(i,:),'-','Color',[0.8 0.8 0.8],'LineWidth',0.8);
            hold on;
        end
        L1 = plot(sPeriod,Sa_design,'-','color','#000000',LineWidth=1.5);
        meanSa = mean(Sa_database);
        L3 = plot(sPeriod,meanSa,'--','color','#A2142F',LineWidth=1.5);
%         plot([0.82 0.82],[0 2],'--','color','#0072BD',LineWidth=1.5);
%         text(0.82,0.05,'{T_1}','Color','#0072BD','FontSize',12);
        hold off;
        legend([L1,L2,L3],{'Target spectrum','Individual sample','Mean spectrum'});
        ylabel('Spectral acceleration, {\itSa} (g)');
        xlabel('Period (s)');
        xlim([0,6])
        ylim([0,2])
        set(gca,'LooseInset',[0.04,0.06,0.02,0.02],'FontName','Times New Roman','FontSize',16,LineWidth=1.5)
end