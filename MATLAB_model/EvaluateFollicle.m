%event function, specifies when the integration has to stop, i.e. when a
%follicle ovulates (growths bigger than max. size m)
%m = ovulation size
function [lookfor,stop,direction] = EvaluateFollicle(t,y,para,parafoll,LH)

m = parafoll(7);
th = t-0.5;
[~, idx] = min(abs(LH.Time-th));
y_lh  = LH.Y(idx);

%number of current follicle(s)
NumFoll=size(y,1)-para(2);
%size(s) of current follicle(s)
FollSize = y(1:NumFoll);

if max(FollSize) >= (m-0.001) &&  y_lh>= parafoll(10)
    %a follicle might ovulate
    lookfor = ((m-0.001) - max(FollSize));
    stop = 1;
    %locates zeros where the event function is decreasing
    direction = -1;
else
    lookfor = 0;
    stop = 0;
    direction = -1;
end

end
