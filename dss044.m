function uxx = dss044(a, b, n, u, ux, nl, nu)
h = (b - a) / (n - 1);
uxx = zeros(size(u));
uxx(2:n-1) = (u(3:n) - 2*u(2:n-1) + u(1:n-2)) / h^2;
if nl == 1, uxx(1) = 0; else uxx(1) = uxx(2); end
if nu == 1, uxx(n) = 0; else uxx(n) = uxx(n-1); end
