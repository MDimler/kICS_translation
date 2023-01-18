# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:28:13 2023

@author: Martin
"""

import numpy as np
import math

def kICSSlideWinFn(s, kSqGrid, tauGrid):
    
    A = s.D * kSqGrid
    
    # static autocorreltaion term
    static_term = (s.K**(-2) * ((-1)+s.r) * s.r * s.Ts**(-2) * ((-1)*math.exp(1)**((-1)*s.K*(
        (-1)+s.Ts+(-1)*tauGrid))*(1+math.exp(1)**((-2)*s.K*tauGrid))*(math.exp(1)**((-1)*s.K)
        +(1+(-1)*math.exp(1)**((-1)*s.K))*s.Ts)+2*(s.K*tauGrid+math.exp(1)**((-1)*s.k*tauGrid)
        *(1+((-1)+s.Ts)*s.Ts+(-1)*((-1)+s.Ts)*s.Ts*math.cosh(s.K)))))
    
    diff_term =(A**(-2)*(A+s.K)**(-2)*s.r*(math.exp(1)**((-1)*A*((-1)+tauGrid))*(1+math.exp(1)**((-2)*A)
        +(-2)*math.exp(1)**((-1)*A))*s.K*(2*A+s.K)*s.r+(-1)*A**2*math.exp(1)**(((-1)*A+(-1)*s.K)
        *((-1)+tauGrid))*((1+(-1)*math.exp(1)**((-1)*A+(-1)*s.K))**2*((-1)+s.r)+(-1)
        *math.exp(1)**((-1)*s.K*(1+(-1)*tauGrid))*(1+(-1)*math.exp(1)**((-1)*A))**2*s.r)+(-1)
        *(math.exp(1)**((-1)*A*((-1)+tauGrid))*(1+math.exp(1)**((-2)*A))*s.K*(2*A+s.K)*s.r
        +(-1)*math.exp(1)**((-1)*A*((-2)+tauGrid)+(-1)*s.K*((-1)+tauGrid))*(A**2*math.exp(1)**((-1)*A)
        *(1+(-1)*math.exp(1)**((-1)*A+(-1)*s.K))*(1+(-1)*math.exp(1)**(((-1)*A+(-1)*s.K)*s.Ts))*((-1)+s.r)
        +math.exp(1)**((-1)*A+(-1)*s.K*(1+(-1)*tauGrid))*((-1)*A**2+math.exp(1)**((-1)*A)*(A+s.K)**2
        +math.exp(1)**((-1)*A*s.Ts)*(1+(-1)*math.exp(1)**((-1)*A))*(A+s.K)**2+math.exp(1)**((-2)*A)*s.K
        *(2*A+s.K))*s.r))*s.Ts**(-1)+((-2)*A**3+math.exp(1)**((-1)*A*tauGrid)*(1+(-1)*math.exp(1)**((-1)*A))
        *(1+math.exp(1)**((-1)*A*((-1)+s.Ts+(-2)*tauGrid)))*s.K**2*s.r+(-2)*A*s.K*((-1)*math.exp(1)**((-1)*A
        *((-1)+s.Ts+(-1)*tauGrid))+math.exp(1)**((-1)*A*(s.Ts+(-1)*tauGrid))+(-1)*math.exp(1)**((-1)*A*tauGrid)
        +math.exp(1)**((-1)*A*(1+tauGrid))+s.K)*s.r+A**2*((-1)*math.exp(1)**(((-1)*A+(-1)*s.K)*((-1)
        +s.Ts+(-1)*tauGrid))*((-1)+s.r)+math.exp(1)**(((-1)*A+(-1)*s.K)*(s.Ts+(-1)*tauGrid))
        *((-1)+s.r)+(-1)*math.exp(1)**(((-1)*A+(-1)*s.K)*tauGrid)*((-1)+s.r)+math.exp(1)**(((-1)*A+(-1)*s.K)
        *(1+tauGrid))*((-1)+s.r)+math.exp(1)**((-1)*A*((-1)+s.Ts+(-1)*tauGrid))*s.r+(-1)*math.exp(1)**((-1)
        *A*(s.Ts+(-1)*tauGrid))*s.r+math.exp(1)**((-1)*A*tauGrid)*s.r+(-1)*math.exp(1)**((-1)*A*(1+tauGrid))
        *s.r+(-2)*s.K*(1+s.r)))*s.Ts**(-1)+s.Ts**(-2)*((-1)*math.exp(1)**((-1)*A*((-1)+tauGrid))*(A**2
        *math.exp(1)**((-1)*s.K*((-1)+tauGrid))*(1+(-2)*math.exp(1)**((-1)*A+(-1)*s.K)+math.exp(1)**((-2)*(A+s.K)))
        *((-1)+s.r)+(-1)*(1+math.exp(1)**((-2)*A)+(-2)*math.exp(1)**((-1)*A))*(A+s.K)**2.*s.r)*s.Ts
        +math.exp(1)**((-1)*A*tauGrid)*((A+s.K)**2*s.r*((-1)+math.exp(1)**((-1)*A*s.Ts)+s.Ts
        +(-1)*math.exp(1)**((-1)*A)*s.Ts)+A**2*math.exp(1)**((-1)*s.K*tauGrid)*((-1)+s.r)
        *(1+(-1)*math.exp(1)**(((-1)*A+(-1)*s.K)*s.Ts)+(-1)*s.Ts+math.exp(1)**((-1)*A+(-1)*s.K)*s.Ts))
        +2*A*(A+s.K)*(A+A*(1+(-1)*math.exp(1)**((-1)*A+(-1)*s.K))*(A+s.K)**(-1)*((-1)+s.r)
        +((-1)+math.exp(1)**((-1)*A))*s.r+A**(-1)*((-1)+A+math.exp(1)**((-1)*A))*s.K*s.r)
        *(s.Ts+(-1)*tauGrid)+2*A*s.K*s.r*((-1)+math.exp(1)((-1)*A*(s.Ts+(-1)*tauGrid))+s.Ts
        +(-1)*math.exp(1)**((-1)*A)*(s.Ts+(-1)*tauGrid)+(-1)*tauGrid)+s.K**2*s.r*((-1)
        +math.exp(1)**((-1)*A*(s.Ts+(-1)*tauGrid))+s.Ts+(-1)*math.exp(1)**((-1)*A)*(s.Ts+(-1)*tauGrid)
        +(-1)*tauGrid)+A**2*((-1)+(-1)*math.exp(1)**(((-1)*A+(-1)*s.K)*(s.Ts+(-1)*tauGrid))
        *((-1)+s.r)+math.exp(1)**((-1)*A*(s.Ts+(-1)*tauGrid))*s.r+s.Ts+math.exp(1)**((-1)*A+(-1)*s.K)
        *((-1)+s.r)*(s.Ts+(-1)*tauGrid)+(-1)*math.exp(1)**((-1)*A)*s.r*(s.Ts+(-1)*tauGrid)+(-1)*tauGrid)
        +(-1)*(A+s.K)**2*s.r*((-1)+(-1)*math.exp(1)**((-1)*A*tauGrid)*((-1)+s.Ts)+(-1)*s.Ts
        +math.exp(1)**((-1)*A*((-1)+tauGrid))*s.Ts+math.exp(1)**((-1)*A)*(s.Ts+(-1)*tauGrid)+tauGrid)
        +A**2*((-1)+s.r)*((-1)+(-1)*math.exp(1)**(((-1)*A+(-1)*s.K)*tauGrid)*((-1)+s.Ts)+(-1)*s.Ts
        +math.exp(1)**(((-1)*A+(-1)*s.K)*((-1)+tauGrid))*s.Ts+math.exp(1)**((-1)*A+(-1)*s.K)
        *(s.Ts+(-1)*tauGrid)+tauGrid))))
                             
    diff_term_norm = (A**(-2)*math.exp(1)**(A+s.K)*(A+s.K)**(-2)*s.r*s.Ts**(-2)*(2*A**3
        *math.exp(1)**((-1)*A+(-1)*s.K)*((-1)+s.Ts)*s.Ts+(1/2)*A**2*(1+(-1)*math.exp(1)**((-2)*(A+s.K)))
        *s.r*s.Ts**2+(1/2)*A**2*(1+math.exp(1)**((-2)*(A+s.K)))*s.Ts*((-4)+4*s.r+2*s.Ts
        +(-3)*s.r*s.Ts)+2*A*math.exp(1)**((-1)*A+(-1)*s.K)*s.K*s.r*((-2)+math.exp(1)**((-1)*A*s.Ts)
        +(math.exp(1)**((-1)*A))**s.Ts+2*math.exp(1)**((-1)*A*((-1)+s.Ts))*(1+(-1)*math.exp(1)**((-1)*A))
        *s.Ts+2*math.exp(1)**((-1)*A)*((-1)+s.Ts)*s.Ts+((-2)+s.K)*((-1)+s.Ts)*s.Ts)+math.exp(1)**((-1)*A
        +(-1)*s.K)*s.K**2*s.r*((-2)+(math.exp(1)**((-1)*A))**s.Ts+2*s.Ts+(-2)*s.Ts*(math.exp(1)**((-1)*A)
        +(1+(-1)*math.exp(1)**((-1)*A))*s.Ts)+math.exp(1)**((-1)*A*((-1)+s.Ts))*(math.exp(1)**((-1)*A)
        +2*(1+(-1)*math.exp(1)**((-1)*A))*s.Ts))+A**2*((-2)*math.exp(1)**((-1)*A+(-1)*s.K)
        +math.exp(1)**(((-1)*A+(-1)*s.K)*(1+s.Ts))+(-2)*math.exp(1)**(((-1)*A+(-1)*s.K)*s.Ts)*((-1)+s.r)
        *s.Ts+2*math.exp(1)**((-1)*s.K+(-1)*A*s.Ts)*s.r*s.Ts+((-1)+s.r)*((-2)+s.Ts)*s.Ts
        +2*math.exp(1)**((-1)*A+(-1)*s.K)*((-1)+s.K)*((-1)+s.Ts)*s.Ts+2*math.exp(1)**((-2)*A+(-1)*s.K)
        *s.r*((-1)+s.Ts)*s.Ts+math.exp(1)**((-2)*(A+s.K))*s.Ts**2+math.exp(1)**(((-1)*A+(-1)*s.K)*(1+s.Ts))
        *((-1)+s.r)*((-1)+2*s.Ts)+math.exp(1)**((-1)*A+(-1)*s.K+(-1)*A*s.Ts)*(s.r+(-2)*s.r*s.Ts)
        +math.exp(1)**((-1)*A+(-1)*s.K)*s.r*(math.exp(1)**((-1)*A*s.Ts)+(-1)*math.exp(1)**(((-1)*A
        +(-1)*s.K)*s.Ts)+2*s.K*((-1)+s.Ts)*s.Ts))))
                                               
    static_term_norm = (2*s.K**(-2)*((-1)+s.r)*s.r*s.Ts**(-2)*(1+math.exp(1)**((-1)*s.K*((-1)+s.Ts))
        *(math.exp(1)**((-1)*s.K)*((-1)+s.Ts)+(-1)*s.Ts)+s.Ts*((-1)+s.K+s.Ts+(-1)*s.K*s.Ts)
        +math.exp(1)**((-1)*s.K)*(s.Ts+(-1)*s.Ts**2)))
    
    return (static_term,diff_term,static_term_norm,diff_term_norm)

