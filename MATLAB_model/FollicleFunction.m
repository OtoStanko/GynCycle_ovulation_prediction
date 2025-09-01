function f=FollicleFunction(t,y,Tovu,Follicles,technicalParameters,follicleParameters,Par,dd1,simulationSettings,firstExtraction)

nnfe = technicalParameters.numNonFollicleEq;
%determine number of active follicles
NumFollicles=size(y,1)-nnfe;

if( NumFollicles > 0 )
    x= y(1:NumFollicles);
else
    x=0;
end

if(NumFollicles > 0  && technicalParameters.shouldTest==0)
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
dy = HormoneModel(t, y, Par, nnfe);
f=dy;
%
%-----------------------------------------------------------------------
%
[r,~] = size(y);

fshrezcomp = y(r-nnfe+3);
p4all = y(r-nnfe+2);
SumV = sum(x.^follicleParameters.fractalDim);

for i = 1:(NumFollicles)

    %FSH sensitivity of the follicles
    fFSH=Follicles.ActiveFSHS(i);
    %fsize = y(i);

    %growth rate
    gamma = follicleParameters.gamma*((1/(1+(p4all/3)^3))+(fshrezcomp^5)/(0.95^5+fshrezcomp^5)); %1 %5

    %negative Hill function for FSH with kappa(proportion of self harm)
    kappa=follicleParameters.k*(0.55^10/(0.55^10+fshrezcomp^10)); %1

    xi=follicleParameters.xi;

    ffsh = (fshrezcomp)^4/(fshrezcomp^4+(fFSH)^4);

    %follicles growth equation
    X = ffsh*(xi-y(i))*y(i)*(gamma-(kappa*(SumV-(follicleParameters.mu*(y(i)^follicleParameters.fractalDim)))));

    if( technicalParameters.shouldTest == 0 )
        %if the size of the foll. is decreasing (or constant),
        %OR the size increases very slow and the follicle is 2 or more days alive
        %OR the foll. is big but alive for two or more days and has not ovulated
        %OR the foll. is more than 20 days alive,
        %then set destiny to decrease (-2)
        %and make it decreasing in size faster
        if X <= follicleParameters.tooSLowFollGrowth || ...
          Follicles.Follicle{Follicles.Active(i)}.Destiny == -2 ||...
          X <= follicleParameters.verySlowFollGrowth && (t - Follicles.Follicle{Follicles.Active(i)}.Time(1)) >= follicleParameters.maxLifeBigRestingFolls && Follicles.Follicle{Follicles.Active(i)}.Destiny == 3 ||...
          X <= follicleParameters.verySlowFollGrowth && (t - Follicles.Follicle{Follicles.Active(i)}.Time(1)) >= follicleParameters.maxLifeSmallGrowingFolls && Follicles.Follicle{Follicles.Active(i)}.Destiny == -1 ||...
          Follicles.Follicle{Follicles.Active(i)}.Destiny == 3 && (t- Follicles.Follicle{Follicles.Active(i)}.TimeDecrease) >= follicleParameters.bigFollLivetime ||...
          (Follicles.Follicle{Follicles.Active(i)}.Time(1) - Follicles.Follicle{Follicles.Active(i)}.Time(end)) > follicleParameters.maxLifeBigRestingFolls
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
%Calculate temperature
f(NumFollicles+18)=f(NumFollicles+18) + 0.02*(y(r-nnfe+2)) + 0.5*((36.5-y(r-nnfe+18)));
%
%-----------------------------------------------------------------------
%
%E2 production
%
%%Calculate follicular surface
%
if(NumFollicles > 0  && technicalParameters.shouldTest==0)
    for i = 1:(NumFollicles)
        if( (Follicles.Follicle{Follicles.Active(i)}.Destiny == 4 ))
            x(i)=0;
        end
    end
end

SF = pi*sum((x.^Par(57))./(x.^Par(57)+Par(58)^Par(57)).*(x.^2));
%
%%calculate E2 concentration
f(NumFollicles+1)=y(NumFollicles+1) - Par(75) - (Par(59) + Par(60)*SF) - Par(61)*exp(-Par(62)*(t-(Tovu+7))^2);
%
%-----------------------------------------------------------------------
%
%Calculation of P4 values
f(NumFollicles+2)=y(NumFollicles+2) - Par(76) - Par(63)*exp(-Par(62)*(t-(Tovu+7))^2);
%
%-----------------------------------------------------------------------
%
%Calculation of FSHAnaC
%
%-----------------------------------------------------------------------
%
if simulationSettings.stim == 0
    f(NumFollicles+17)=y(NumFollicles+17)-0;
    f(NumFollicles+16)=y(NumFollicles+16)-0;
end
%
switch simulationSettings.simulationType
    case SimulationType.LutStim
    if Par(64) > 0 && t > Par(71)
         n = dd1;
         H = 0;
         J = 0;
         for i = 1:n
             dt = Par(71) + i - 1 ;
             h  = ((Par(65)*(Par(66)^2))/((Par(66)-Par(67))^2)) ...
                        * [exp(-Par(66)*(t-dt)) * (Par(67)*(t-dt) ...
                        -Par(66)*(t-dt)-1)+exp(-Par(67)*(t-dt))]; %#ok<*NBRAK1>
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
    case SimulationType.FollStim
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
    case SimulationType.DoubleStim
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
