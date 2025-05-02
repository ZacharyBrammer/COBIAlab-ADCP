function adcp=read_vadcp(fname)

%start by getting first data batch
CHECK = 0;
tic;
fid = fopen(fname);
bufferSize = 1e6;
eoe = 127;

dataBatch = fread(fid,bufferSize,'uint8');

dataIncrement = fread(fid,2,'uint8');

while ~isempty(dataIncrement) && dataIncrement(end) ~= eoe && dataIncrement(end-1) ~= eoe && ~feof(fid)
    dataIncrement(end+1) = fread(fid,1,'uint8');  %This can be slightly optimized
end

data = [dataBatch; dataIncrement(1:end-1)];

cum_index=0;
trip=0;

while ~isempty(data)
    
test=dec2hex(data);
tt=cellstr(test);
I=find(strcmp(tt,'7F'));
J= diff(I)==1;
add_index=cum_index+1+I(J);
if ~exist('ens_index')
    ens_index=I(J);
else
    ens_index=[ens_index; add_index];
end

cum_index=cum_index+length(data);

disp([num2str(length(data)),' bytes scanned with ',num2str(length(I(J))),' ensembles.']);

clear I J tt test add_index;

dataBatch = fread(fid,bufferSize,'uint8');

dataIncrement = fread(fid,2,'uint8');

while ~isempty(dataIncrement) && dataIncrement(end-1) ~= eoe && dataIncrement(end) ~= eoe && ~feof(fid)
    dataIncrement(end+1) = fread(fid,1,'uint8');  %This can be slightly optimized
end

data = [dataBatch; dataIncrement];

%trip=trip+1

end

%%

disp([num2str(length(ens_index)),' ensembles found...processing...']);

fseek(fid,0,'bof');

for ii=1:length(ens_index)-1
    
    
    ens_offset=ens_index(ii)-1;   
    
    if rem(ii,1000)==0
        disp([num2str(floor(ii./length(ens_index).*100)), '% done...']);
    end
    
    
%header data
    offset=ens_offset;
    fseek(fid,offset,'bof');
    cfgid=fread(fid,2,'uint8');
    %fprintf('%s%d%s%s%s\n','index = ',ii,', ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
    hdr.numbytes(ii)=fread(fid,1,'int16');
            
    cmpt=0;
    
    if ii<=1
        cmpt=1;
        test=hdr.numbytes(ii);
    elseif (hdr.numbytes(ii)>=test-10 && hdr.numbytes(ii)<=test+10)
        cmpt=1;
    end
    
    if cmpt==1
    
    fseek(fid,1,'cof');
    hdr.datatypes=fread(fid,1,'int8');
    hdr.dat_offsets=fread(fid,hdr.datatypes,'int16');
    
%       for ii=1:length(hdr.dat_offsets)
%           hdr.dat_offsets(ii)
%       end
    
%configuration data

    offset=ens_offset+hdr.dat_offsets(1);
    fseek(fid,offset,'bof');
    cfgid=fread(fid,2,'uint8');
    %fprintf('%s%s%s\n','ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
    cfg.name           ='vadcp';
    cfg.sourceprog     ='instrument';
    cfg.prog_ver       =fread(fid,1,'uint8')+fread(fid,1,'uint8')/100; 
    config             =fread(fid,2,'uint8');  % Coded stuff
    cfg.config         =[dec2base(config(2),2,8) '-' dec2base(config(1),2,8)];
%     cfg.beam_angle     =getopt(bitand(config(2),3),15,20,30);
%     cfg.numbeams       =getopt(bitand(config(2),16)==16,4,5);
%     cfg.beam_freq      =getopt(bitand(config(1),7),75,150,300,600,1200,2400,38);
%     cfg.beam_pattern   =getopt(bitand(config(1),8)==8,'concave','convex'); % 1=convex,0=concave
%     cfg.orientation    =getopt(bitand(config(1),128)==128,'down','up');    % 1=up,0=down
    fseek(fid,2,'cof');
    cfg.n_beams        =fread(fid,1,'uint8');
    cfg.n_cells        =fread(fid,1,'uint8');
    cfg.pings_per_ensemble=fread(fid,1,'uint16');
    cfg.cell_size      =fread(fid,1,'uint16')*.01;	 % meters
    cfg.blank          =fread(fid,1,'uint16')*.01;	 % meters
    fseek(fid,1,'cof');          %
    cfg.corr_threshold =fread(fid,1,'uint8');
    cfg.n_codereps     =fread(fid,1,'uint8');
    fseek(fid,1,'cof');
    cfg.evel_threshold =fread(fid,1,'uint16');
    cfg.time_between_ping_groups=sum(fread(fid,3,'uint8').*[60 1 .01]'); % seconds
    coord_sys      =fread(fid,1,'uint8');                                % Lots of bit-mapped info
      cfg.coord=dec2base(coord_sys,2,8);
%       cfg.coord_sys      =getopt(bitand(bitshift(coord_sys,-3),3),'beam','instrument','ship','earth');
%       cfg.use_pitchroll  =getopt(bitand(coord_sys,4)==4,'no','yes');  
%       cfg.use_3beam      =getopt(bitand(coord_sys,2)==2,'no','yes');
%       cfg.bin_mapping    =getopt(bitand(coord_sys,1)==1,'no','yes');
    fseek(fid,4,'cof');
    cfg.sensors_src    =dec2base(fread(fid,1,'uint8'),2,8);
    cfg.sensors_avail  =dec2base(fread(fid,1,'uint8'),2,8);
    cfg.bin1_dist      =fread(fid,1,'uint16')*.01;	% meters
    fseek(fid,4,'cof');
    cfg.fls_target_threshold =fread(fid,1,'uint8');
    fseek(fid,1,'cof');
    cfg.xmit_lag       =fread(fid,1,'uint16')*.01; % meters
    fseek(fid,8,'cof');
    cfg.bandwidth     =fread(fid,1,'uint16');
    cfg.syspower     =fread(fid,1,'uint8');
    fseek(fid,1,'cof');
    cfg.sernum     =fread(fid,1,'uint32');
    cfg.b_angle     =fread(fid,1,'uint8');
    
    cfg.ranges=cfg.bin1_dist+[0:(cfg.n_cells-1)]'*cfg.cell_size;
 %   if cfg.orientation==1, cfg.ranges=-cfg.ranges; end   
    
%system data
    offset=ens_offset+hdr.dat_offsets(2);
    fseek(fid,offset,'bof');
    cfgid=fread(fid,2,'uint8');
    %fprintf('%s%s%s\n','ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
    ens.number(ii)         =fread(fid,1,'uint16');
    rtc=fread(fid,7,'uint8');
    ens.mtime(ii)=datenum(datenum(rtc(1)+2000,rtc(2),rtc(3),rtc(4),rtc(5),rtc(6)));
    fseek(fid,5,'cof');
    ens.depth(ii)          =fread(fid,1,'uint16')*.1;   % meters
    fseek(fid,6,'cof');
    ens.salinity(ii)       =fread(fid,1,'int16');       % PSU
    ens.temperature(ii)    =fread(fid,1,'int16')*.01;   % Deg C
    ens.mpt(ii)            =sum(fread(fid,3,'uint8').*[60 1 .01]'); % seconds
    fseek(fid,9,'cof');
    ens.voltage(ii)        =fread(fid,1,'int8').*157/1000;
    
%velocity data
     offset=ens_offset+hdr.dat_offsets(3);
     fseek(fid,offset,'bof');
     cfgid=fread(fid,2,'uint8');
     %fprintf('%s%s%s\n','ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
     vels=fread(fid,[4 cfg.n_cells],'int16')'*.001;     % m/s
     ens.x_vel(:,ii)  =vels(:,1);
     ens.y_vel(:,ii)  =vels(:,2);
     ens.z_vel(:,ii)  =vels(:,3);
     
%correlations
     offset=ens_offset+hdr.dat_offsets(5);
     fseek(fid,offset,'bof');
     cfgid=fread(fid,2,'uint8');
     %fprintf('%s%s%s\n','ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
     temp_corr   =fread(fid,[4 cfg.n_cells],'uint8')';
     ens.corr(:,:,ii)    =temp_corr(:,1:3);
     fseek(fid,1,'cof');

%echo intensity
     offset=ens_offset+hdr.dat_offsets(6);
     fseek(fid,offset,'bof');
     cfgid=fread(fid,2,'uint8');
     %fprintf('%s%s%s\n','ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
     temp_intens   =fread(fid,[4 cfg.n_cells],'uint8')';
     ens.intens(:,:,ii)    =temp_intens(:,1:3);
     
%percent good
     offset=hdr.dat_offsets(7);
     fseek(fid,offset,'bof');
     cfgid=fread(fid,2,'uint8');
     %fprintf('%s%s%s\n','ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
     temp_pg   =fread(fid,[4 cfg.n_cells],'uint8')';
     ens.perc_good(:,:,ii)    =temp_pg(:,1:3);
     
%surface track status output
     offset=ens_offset+hdr.dat_offsets(11);
     fseek(fid,offset,'bof');
     cfgid=fread(fid,2,'uint8');
     %fprintf('%s%s%s\n','ID : ', dec2hex(cfgid(2),2),dec2hex(cfgid(1),2));
     ens.surface_track(ii)=fread(fid,1,'uint32').*.0001; %in m
     ens.surface_track_uncorr(ii)=fread(fid,1,'uint32').*.0001; %in m
     fseek(fid,1,'cof');
     ens.v_amp(ii)=fread(fid,1,'uint8'); %in counts
     ens.v_pgood(ii)=fread(fid,1,'uint8'); %in counts
    
   else
    
     ens.number(ii)         =NaN;
     ens.mtime(ii)          =NaN;
     ens.depth(ii)          =NaN;
     ens.salinity(ii)       =NaN;
     ens.temperature(ii)    =NaN;
     ens.mpt(ii)            =NaN;
     ens.pressure(ii)       =NaN;
     ens.x_vel(:,ii)        =NaN;
     ens.y_vel(:,ii)        =NaN;
     ens.z_vel(:,ii)        =NaN;
     ens.corr(:,:,ii)       =NaN;
     ens.intens(:,:,ii)     =NaN;    
    end
end


adcp=ens;
adcp.config=cfg;

toc

end


%-------------------------------------
function opt=getopt(val,varargin)
% Returns one of a list (0=first in varargin, etc.)
if val+1>length(varargin),
	opt='unknown';
else
   opt=varargin{val+1};
end;
end