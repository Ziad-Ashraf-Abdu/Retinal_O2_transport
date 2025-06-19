% pde_1_main.m
clear all
clc

global nir nor nfl ncc ...
       zl_ir zl_or zl_fl zl_cc ...
       zg_ir zg_or zg_fl zg_cc ...
       Dir Dor Dfl Dcc ...
       kir_or kor_fl kfl_cc ...
       kir kor kfl kcc ...
       pir_s pcc_s ncall ncase

ncase=2;
nir=11; nor=11; nfl=11; ncc=11;
zl_ir=200; zl_or=200; zl_fl=200; zl_cc=200;
zg_ir=[0:zl_ir/10:zl_ir]'; zg_or=[0:zl_or/10:zl_or]';
zg_fl=[0:zl_fl/10:zl_fl]'; zg_cc=[0:zl_cc/10:zl_cc]';

Dir=1.0e+04; Dor=1.0e+04; Dfl=1.0e+04; Dcc=1.0e+04;
kir_or=1; kor_fl=1; kfl_cc=1;

if(ncase==1)
    kir=0; kor=0; kfl=0; kcc=0;
elseif(ncase==2)
    kir=0.1; kor=0.1; kfl=0.1; kcc=0.1;
end

pir_s=20; pcc_s=100;
t0=0.0; tf=30.0; tout=linspace(t0, tf, 7)';
nout=length(tout);

u0 = inital_1(t0);

mf=2;
reltol=1.0e-6; abstol=1.0e-5;
options = odeset('RelTol', reltol, 'AbsTol', abstol);

if mf==2
    S = jpattern_num_1;
    options = odeset(options, 'JPattern', S);
    [t, u] = ode15s(@pde_1, tout, u0, options);
end

for it=1:nout
    for i=1:nir, uir(it,i) = u(it,i); end
    for i=1:nor, uor(it,i) = u(it,i+nir); end
    for i=1:nfl, ufl(it,i) = u(it,i+nir+nor); end
    for i=1:ncc, ucc(it,i) = u(it,i+nir+nor+nfl); end
    if(it>=2)
        uir(it,1) = pir_s;
        ucc(it,ncc) = pcc_s;
    end
end

figure(1)
subplot(2,2,1)
plot(zg_ir, uir, '-')
xlabel('zg-ir, \mum'); ylabel('uir(z,t), mm Hg O_2');
title('Inner Retina'); axis([0 200 0 100])

subplot(2,2,2)
plot(zg_or, uor, '-')
xlabel('zg-or, \mum'); ylabel('uor(z,t), mm Hg O_2');
title('Outer Retina'); axis([0 200 0 100])

subplot(2,2,3)
plot(zg_fl, ufl, '-')
xlabel('zg-fl, \mum'); ylabel('ufl(z,t), mm Hg O_2');
title('Fluid Layer'); axis([0 200 0 100])

subplot(2,2,4)
plot(zg_cc, ucc, '-')
xlabel('zg-cc, \mum'); ylabel('ucc(z,t), mm Hg O_2');
title('Choroid'); axis([0 200 0 100])
