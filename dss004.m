function ux = dss004(a, b, n, u)
h = (b - a) / (n - 1);
ux = zeros(size(u));
ux(2:n-1) = (u(3:n) - u(1:n-2)) / (2*h);
ux(1) = (u(2) - u(1)) / h;
ux(n) = (u(n) - u(n-1)) / h;
