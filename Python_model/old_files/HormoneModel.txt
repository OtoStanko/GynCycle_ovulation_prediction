function dy = ODE_Model_NormalCycle(t,y,Par)
%
%-----------------------------------------------------------------------
%
[r,c] = size(y);
dy=zeros(r,c);
%
%-----------------------------------------------------------------------
%
i_FSH_med  = r;
i_LH_med   = r-1;

i_GnRH     =  r-2;
i_RecGa    =  r-3;
i_RecGi    =  r-4;
i_GReca    =  r-5;
i_GReci    =  r-6; 

i_RP_LH    =  r-7;
i_LH       =  r-8;

i_RP_FSH   =  r-9;
i_FSH      =  r-10;
i_FSHfoll  =  r-11;
i_RFSH     =  r-12;
i_RFSH_des =  r-13;
i_FSHR     =  r-14;

i_P4       =  r-15;
i_E2       =  r-16;
%
%-----------------------------------------------------------------------
%
%GnRH
%%GnRH frequency and mass 
%
  yGfreq = Par(1) / ( 1 + ( y(i_P4) / Par(2) ) ^ Par(3) ) ...
              * ( 1 + y(i_E2) ^ Par(4) ...
              / ( Par(5) ^ Par(4) + y(i_E2) ^ Par(4) ) );
%
  yGmass = Par(6) * (y(i_E2) ^ Par(7) / ( Par(8) ^ Par(7) + y(i_E2) ^ Par(7) ) ...
              + Par(9) ^ Par(10) / ( Par(9) ^ Par(10) + y(i_E2) ^ Par(10) ));
%
%%GnRH in pituitary 
%
  dy(i_GnRH) =   yGmass * yGfreq ... 
               - Par(11) * y(i_GnRH) * y(i_RecGa) ...
               + Par(12) * y(i_GReca) ...
               - Par(13) * y(i_GnRH);
%
%%GnRH receptor mechanisms
%
%%%active GnRH receptor
%
  dy(i_RecGa) =   Par(12) * y(i_GReca) ...
               - Par(11) * y(i_GnRH) * y(i_RecGa) ...  
               - Par(14) * y(i_RecGa) ...
                + Par(15) * y(i_RecGi);
%
%%%inactive GnRH receptor	 
%
  dy(i_RecGi) =   Par(16) ...
                + Par(14) * y(i_RecGa) ...
                - Par(15) * y(i_RecGi) ...
                + Par(17) * y(i_GReci) ...
                - Par(18) * y(i_RecGi);
%
% active GnRH-receptor complex	 
%
  dy(i_GReca) =   Par(11) * y(i_GnRH) * y(i_RecGa) ...
                - Par(12) * y(i_GReca) ...
                - Par(19) * y(i_GReca) ...
                + Par(20) * y(i_GReci);
%
% inactive GnRH-receptor complex
%
  dy(i_GReci) =   Par(19) * y(i_GReca) ... 
                - Par(20) * y(i_GReci) ...
                - Par(17) * y(i_GReci) ...
                - Par(21) * y(i_GReci);
%
%-----------------------------------------------------------------------
%
%LH
%
%%LH in pituitary
%
%%%LH production
%
  hp_e2      = 1.5 * ( y(i_E2) / Par(22) ) ^ Par(23) ... 
                / ( 1 + ( y(i_E2) / Par(22) ) ^ Par(23) );
  hp_freq    = ((yGfreq / Par(24)) ^ Par(25)) / (1 + (yGfreq / Par(24))^Par(25));
  f_LH_prod1 = Par(26) + Par(27) * hp_e2; 
  hm_p4      = 1 + Par(30)*( y(i_P4) / Par(28) ) ^ Par(29); 
  
  f_LH_prod  = (f_LH_prod1 / hm_p4 ) * (1 + hp_freq);
%
%%%LH release
%
  f_LH_rel   = ( Par(34) + Par(35) * ... 
               ( ( y(i_GReca)) / Par(36) ) ^ Par(37) ...
             / ( 1 + ( ( y(i_GReca) ) ...
               / Par(36) ) ^ Par(37) ) ); 
           
  dy(i_RP_LH) = f_LH_prod - f_LH_rel * y(i_RP_LH);
%
%%LH in the blood
%     
  dy(i_LH)   =   ( 1 / Par(38)) * f_LH_rel * y(i_RP_LH)...
               - Par(39) * y(i_LH);
%
%-----------------------------------------------------------------------
%
%FSH
%
%%FSH pituitary
%
%%%FSH production
%
  f_FSH_prod1   = Par(40);
  f_FSH_prod2   = 1 + (y(i_P4)/Par(41)) ^ Par(42);
  hm_freq       = 1 / ( 1 + ( yGfreq / Par(43) ) ^ Par(44) );

  f_FSH_prod    = (f_FSH_prod1 / f_FSH_prod2) *  hm_freq;  
%
  f_FSH_rel     =   Par(45) + Par(46) ...
                  *(Par(31) * (( ( y(i_GReca) ) / Par(47) ) ^ Par(48) ...
                  / ( 1 + ( ( y(i_GReca) ) / Par(47) ) ^ Par(48) ))) ...
                  * 1/(1 + (y(i_E2)/Par(73))^Par(74));                   
%     
   dy(i_RP_FSH) = f_FSH_prod - f_FSH_rel * y(i_RP_FSH);
%
%%FSH blood
%      
   dy(i_FSH)    =   1 / Par(38) * f_FSH_rel * y(i_RP_FSH) ...  
                  - Par(50) *  y(i_FSH) ...
                  - Par(51) *  y(i_FSH) ;
%
%%FSH ovaries
%
   dy(i_FSHfoll)=  Par(50) * (y(i_FSH) + y(i_FSH_med)) * Par(52)...
                  - Par(53) * y(i_FSHfoll) * y(i_RFSH) ...
                  - Par(54) * y(i_FSHfoll);

%
%%FSH receptor mechanisms
%
%%%FSH free receptors  
%
   dy(i_RFSH)   =   Par(55) * y(i_RFSH_des) ...
                  - Par(53) * y(i_FSHfoll) * y(i_RFSH);
%
%%%bound FSH receptors
%
  dy(i_FSHR)    =   Par(53) * y(i_FSHfoll) * y(i_RFSH) ...
                  - Par(56) * y(i_FSHR);
%
%%%desensitized FSH receptors
%
  dy(i_RFSH_des) =  Par(56) * y(i_FSHR) ...
                  - Par(55) * y(i_RFSH_des); 
%
%-----------------------------------------------------------------------
%
end