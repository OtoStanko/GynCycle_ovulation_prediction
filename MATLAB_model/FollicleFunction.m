function f=FollicleFunction(t,y,Tovu,Follicles,para,parafoll,Par,dd1,Stim,LutStim,FollStim,DoubStim,firstExtraction)

%determine number of active follicles
NumFollicles=size(y,1)-para(2);

if( NumFollicles > 0 )
    x= y(1:NumFollicles);
else
    x=0;
end

if(NumFollicles > 0  && para(1)==0)
    for i = 1:(NumFollicles)
        if Follicles.Follicle{Follicles.Active(i)}.Destiny == -2 || ...
           Follicles.Follicle{Follicles.Active(i)}.Destiny == -3
            x(i)=0;
        end
    end
end
%
%-----------------------------------------------------------------------
%
% solve differential equations
%
dy = HormoneModel(t, y, Par);
f=dy;
%
%-----------------------------------------------------------------------
%
[r,c] = size(y);

fshrezcomp = y(r-14);
p4all = y(r-15);
SumV = sum(x.^parafoll(1));

for i = 1:(NumFollicles)

    %FSH sensitivity of the follicles
    fFSH=Follicles.ActiveFSHS(i);
    fsize = y(i);

    %growth rate
    gamma = parafoll(2)*((1/(1+(p4all/3)^3))+(fshrezcomp^5)/(0.95^5+fshrezcomp^5)); %1 %5

    %negative Hill function for FSH with kappa(proportion of self harm)
    kappa=parafoll(5)*(0.55^10/(0.55^10+fshrezcomp^10)); %1

    xi=parafoll(3);

    ffsh = (fshrezcomp)^4/(fshrezcomp^4+(fFSH)^4);

    %follicles growth equation
    X = ffsh*(xi-y(i))*y(i)*(gamma-(kappa*(SumV-(parafoll(4)*(y(i)^parafoll(1))))));

    if (para(1)== 1)
        if(X<=0)
            NoFoll=X;
        end
    end

    if( para(1) == 0 )
        %if the size of the foll. is decreasing (or constant),
        %OR the size increases very slow and the follicle is 2 or more days alive
        %OR the foll. is big but alive for two or more days and has not ovulated
        %OR the foll. is more than 20 days alive,
        %then set destiny to decrease (-2)
        %and make it decreasing in size faster
        if X <= parafoll(12) || ...
          Follicles.Follicle{Follicles.Active(i)}.Destiny == -2 ||...
          X <= parafoll(13) && (t - Follicles.Follicle{Follicles.Active(i)}.Time(1)) >= parafoll(15) && Follicles.Follicle{Follicles.Active(i)}.Destiny == 3 ||...
          X <= parafoll(13) && (t - Follicles.Follicle{Follicles.Active(i)}.Time(1)) >= parafoll(14) && Follicles.Follicle{Follicles.Active(i)}.Destiny == -1 ||...
          Follicles.Follicle{Follicles.Active(i)}.Destiny == 3 && (t- Follicles.Follicle{Follicles.Active(i)}.TimeDecrease) >= parafoll(11) ||...
          (Follicles.Follicle{Follicles.Active(i)}.Time(1) - Follicles.Follicle{Follicles.Active(i)}.Time(end)) > parafoll(15)
            %set time the follicle starts to decrease & set destiny to decrease
            if Follicles.Follicle{Follicles.Active(i)}.Destiny ~= -2
                Follicles.Follicle{Follicles.Active(i)}.Destiny = -2;
                Follicles.Follicle{Follicles.Active(i)}.TimeDecrease = t;
            end
            %to decrease the size of the follicle faster
            f(i)= -0.05*y(i)*(t-Follicles.Follicle{Follicles.Active(i)}.TimeDecrease);
            elseif  Follicles.Follicle{Follicles.Active(i)}.Destiny == -3
             f(i) = -1000*y(i);
        else
            %if not dying use normal equation
            f(i)=X;
        end

    else
        %if called to test use normal equation
        f(i)=X;
    end

end
%
%-----------------------------------------------------------------------
%
%E2 production
%
%%Calculate follicular surface
%
if(NumFollicles > 0  && para(1)==0)
    for i = 1:(NumFollicles)
        if( (Follicles.Follicle{Follicles.Active(i)}.Destiny == 4 ))
            x(i)=0;
        end
    end
end

SF = pi*sum((x.^Par(57))./(x.^Par(57)+Par(58)^Par(57)).*(x.^2));
%
%%calculate E2 concentration
%
f(NumFollicles+1)=y(NumFollicles+1) - Par(75) - (Par(59) + Par(60)*SF) - Par(61)*exp(-Par(62)*(t-(Tovu+7))^2);
%
%-----------------------------------------------------------------------
%
%Calculation of P4 values
f(NumFollicles+2)=y(NumFollicles+2)- Par(76) - Par(63)*exp(-Par(62)*(t-(Tovu+7))^2);
%
%-----------------------------------------------------------------------
%
%Calculation of FSHAnaC
%
%-----------------------------------------------------------------------
%
if Stim == 0
    f(NumFollicles+17)=y(NumFollicles+17)-0;
    f(NumFollicles+16)=y(NumFollicles+16)-0;
end
%
if (LutStim)
    if Par(64) > 0 && t > Par(71)
         n = dd1;
         H = 0;
         J = 0;
         for i = 1:n
             dt = Par(71) + i - 1 ;
             h  = ((Par(65)*(Par(66)^2))/((Par(66)-Par(67))^2)) ...
                        * [exp(-Par(66)*(t-dt)) * (Par(67)*(t-dt) ...
                        -Par(66)*(t-dt)-1)+exp(-Par(67)*(t-dt))];
             H  = H + h;

             j  = ((Par(68)*(Par(69)^2))/((Par(69)-Par(70))^2)) ...
                    * [exp(-Par(69)*(t-dt)) * (Par(70)*(t-dt) ...
                    -Par(69)*(t-dt)-1)+exp(-Par(70)*(t-dt))];
             J  = J+j;
         end
        f(NumFollicles+17)=y(NumFollicles+17)-H;
        f(NumFollicles+16)=y(NumFollicles+16)-J;
    else
        f(NumFollicles+17)=y(NumFollicles+17)-0;
        f(NumFollicles+16)=y(NumFollicles+16)-0;
    end
end
%
if (FollStim)
    if Par(64) > 0 && t > Par(71)
         n = dd1;
         s = (-1)^(dd1);
         H = 0;
         J = 0;
         if n < 6
             if s == 1
                for i = 1:n
                     dt = Par(71) + i - 1 ;
                     h  = ((Par(65)*(Par(66)^2))/((Par(66)-Par(67))^2)) ...
                                * [exp(-Par(66)*(t-dt)) * (Par(67)*(t-dt) ...
                                -Par(66)*(t-dt)-1)+exp(-Par(67)*(t-dt))];
                     H  = H + h;

                     j  = ((Par(68)*(Par(69)^2))/((Par(69)-Par(70))^2)) ...
                            * [exp(-Par(69)*(t-dt)) * (Par(70)*(t-dt) ...
                            -Par(69)*(t-dt)-1)+exp(-Par(70)*(t-dt))];
                     J  = J+j;
                end
             else
                 for i = 1:n
                     dt = Par(71) + i - 1 ;
                     h  = ((Par(65)*(2/3)*(Par(66)^2))/((Par(66)-Par(67))^2)) ...
                                * [exp(-Par(66)*(t-dt)) * (Par(67)*(t-dt) ...
                                -Par(66)*(t-dt)-1)+exp(-Par(67)*(t-dt))];
                     H  = H + h;

                     j  = ((Par(68)*(2/3)*(Par(69)^2))/((Par(69)-Par(70))^2)) ...
                            * [exp(-Par(69)*(t-dt)) * (Par(70)*(t-dt) ...
                            -Par(69)*(t-dt)-1)+exp(-Par(70)*(t-dt))];
                     J  = J+j;
                 end
             end
         else
             for i = 1:n
                 dt = Par(71) + i - 1 ;
                 h  = ((Par(65)*(Par(66)^2))/((Par(66)-Par(67))^2)) ...
                            * [exp(-Par(66)*(t-dt)) * (Par(67)*(t-dt) ...
                            -Par(66)*(t-dt)-1)+exp(-Par(67)*(t-dt))];
                 H  = H + h;

                 j  = ((Par(68)*(Par(69)^2))/((Par(69)-Par(70))^2)) ...
                        * [exp(-Par(69)*(t-dt)) * (Par(70)*(t-dt) ...
                        -Par(69)*(t-dt)-1)+exp(-Par(70)*(t-dt))];
                 J  = J+j;
             end
        end
        f(NumFollicles+17)=y(NumFollicles+17)-H;
        f(NumFollicles+16)=y(NumFollicles+16)-J;
    else
        f(NumFollicles+17)=y(NumFollicles+17)-0;
        f(NumFollicles+16)=y(NumFollicles+16)-0;

    end
end
%
if (DoubStim)
    if Par(64) > 0 && t > Par(71)
        n = dd1;
        H = 0;
        J = 0;
        if firstExtraction
            for i = 1:n
                 dt = Par(71) + i - 1 ;
                 h  = ((Par(65)*(Par(66)^2))/((Par(66)-Par(67))^2)) ...
                            * [exp(-Par(66)*(t-dt)) * (Par(67)*(t-dt) ...
                            -Par(66)*(t-dt)-1)+exp(-Par(67)*(t-dt))];
                 H  = H + h;

                 j  = ((Par(68)*(Par(69)^2))/((Par(69)-Par(70))^2)) ...
                        * [exp(-Par(69)*(t-dt)) * (Par(70)*(t-dt) ...
                        -Par(69)*(t-dt)-1)+exp(-Par(70)*(t-dt))];
                 J  = J+j;
            end
        else
            s = (-1)^(dd1);
            if s == -1
                for i = 1:n
                    dt = Par(71) + i - 1 ;
                    h  = ((Par(65)*(2/3)*(Par(66)^2))/((Par(66)-Par(67))^2)) ...
                            * [exp(-Par(66)*(t-dt)) * (Par(67)*(t-dt) ...
                            -Par(66)*(t-dt)-1)+exp(-Par(67)*(t-dt))];
                    H  = H + h;

                    j  = ((Par(68)*(2/3)*(Par(69)^2))/((Par(69)-Par(70))^2)) ...
                        * [exp(-Par(69)*(t-dt)) * (Par(70)*(t-dt) ...
                        -Par(69)*(t-dt)-1)+exp(-Par(70)*(t-dt))];
                    J  = J+j;
                end
            else
                H = 0;
                J = 0;
            end
        end
            f(NumFollicles+17)=y(NumFollicles+17)-H;
            f(NumFollicles+16)=y(NumFollicles+16)-J;
    else
            f(NumFollicles+17)=y(NumFollicles+17)-0;
            f(NumFollicles+16)=y(NumFollicles+16)-0;
    end
end
%
%-----------------------------------------------------------------------
%
end
