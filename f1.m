function [F1] = f1(yactual,ypredict)
a = 0;
b = 0;
c = 0;
d = 0;
for i=1:1:length(yactual)
    if yactual(i) ==1
        if ypredict(i) ==1
            a = a+ 1;
        else
            b = b + 1;
        end
    else
        if ypredict(i) == 1
            c = c + 1;
        else
            d = d + 1;
        end
    end
end
p = a/(a+c);
r = a/(a+b);
F1 = (2*p*r)/(p+r);
end