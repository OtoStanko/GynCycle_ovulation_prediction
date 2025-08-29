function FSHVec = CreateFollicles(follicleParameters,poissonDistributionParameters,tb,te)

%create normal distributed fsh sensitivities for each foll
fileID2 = fopen('FSH.txt','w+');
fprintf(fileID2,'Number    FSH\n');
FSHdistri = makedist('Normal','mu',follicleParameters.meanFSHSensitivity,'sigma',follicleParameters.stdFSHSensitivity);
for i=1:10000
    fsh = random(FSHdistri);
    fprintf(fileID2,'%f %f\n',i,fsh);
end
fclose(fileID2);


%create poisson distributed starting times for each foll
fileID = fopen('StartTimesPoiss.txt','w+');
fprintf(fileID,'Start time\n');
timevec=poissonproc(poissonDistributionParameters.lambda,[tb,te]); 
arraysize=length(timevec);
for i=1:arraysize
    fprintf(fileID,'%f \n',timevec(i));
end
fclose(fileID);


%load StartNumbers and FSH Sensitivities from File
file2 = 'FSH.txt';
delimiterIn=' ';
headerlinesIn=1;
data2=importdata(file2,delimiterIn,headerlinesIn);
NumValFSH = size(data2.data(1:end,1));
NumValFSH = max(NumValFSH);
FSHVec = zeros(NumValFSH,1);
for i = 1:NumValFSH
    FSHVec(i) = data2.data(i,2);
end

end
