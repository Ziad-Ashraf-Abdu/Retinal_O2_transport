function S = jpattern_num_1
global nir nor nfl ncc

ybase = 0.5 * ones(nir+nor+nfl+ncc, 1);
tbase = 0;
ytbase = pde_1(tbase, ybase);
fac = [];
thresh = 1e-16;
vectorized = 'on';
[Jac, fac] = numjac(@pde_1, tbase, ybase, ytbase, thresh, fac, vectorized);
S = sparse(Jac ~= 0);
