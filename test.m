clc; clear;
%% ================= SYMBOLIC VARIABLES =================
syms th2 th3 th4 th5 th6 dth5 dth6 ddth5 ddth6 real
syms r4 r5 r6 m4 m5 m6 l4 l5 l6 g real
q  = [th5; th6];
dq = [dth5; dth6];

%% ================= COM POSITIONS =================
p5 = [ (l5/2)*cos(th5);
       (l5/2)*sin(th5)];

p6 = [ (l5 + l6/2)*cos(th5);
       (l5 + l6/2)*sin(th5) ];

hyp_5=  ((l5*sin(th5)/2)^2+l4^2)^0.5;
hyp_6=  (((l5+l6/2)*sin(th5))^2+l4^2)^0.5;
alpha_5 = atan2(l5/2,l4);
alpha_6 = atan2(l5+l6/2,l4);
beta_5 = -(th2+th3+th4)-alpha_5;
beta_6 = -(th2+th3+th4)-alpha_6;
p5_v = hyp_5*sin(beta_5);
p6_v = hyp_6*sin(beta_6);

%% ================= VELOCITIES =================
v5 = jacobian(p5,q)*dq;
v6 = jacobian(p6,q)*dq;

v5_2 = simplify(v5.'*v5);
v6_2 = simplify(v6.'*v6);

%% ================= INERTIAS =================
I4   = 0.5*m4*r4^2;                  % upstream twist inertia
I5   = (1/12)*m5*(3*r5^2 + l5^2);    % link-5 bending
I6   = (1/12)*m6*(3*r6^2 + l6^2);    % link-6 bending
I6_t = 0.5*m6*r6^2;                  % link-6 twist

%% ================= KINETIC ENERGY =================
T =               0.5*I4*dth5^2 + ...       % link-4 twist
    0.5*m5*v5_2 + 0.5*I5*dth5^2 + ...       % link-5
    0.5*m6*v6_2 + 0.5*I6*dth5^2 + ...       % link-6
                  0.5*I6_t*(dth5 + dth6)^2; % link-6 twist

T = simplify(T);

%% =============== POTENTIAL =======================

V = g * (m5*p5_v + ...
         m6*p6_v);
V = simplify(V);
%% ================= MASS MATRIX =================
M = sym(zeros(2,2));
for i = 1:2
    for j = 1:2
        M(i,j) = simplify(diff(diff(T,dq(i)),dq(j)));
    end
end

%% ================= CORIOLIS MATRIX =================
C = sym(zeros(2,2));
for i = 1:2
    for j = 1:2
        for k = 1:2
            C(i,j) = C(i,j) + 1/2 * ...
                ( diff(M(i,j),q(k)) ...
                + diff(M(i,k),q(j)) ...
                - diff(M(j,k),q(i)) ) * dq(k);
        end
        C(i,j) = simplify(C(i,j));
    end
end

%% ================= GRAVITY =================
G = simplify(jacobian(V,q).');%sym([0;0]);   % horizontal wrist

%% ================= DISPLAY =================
disp('--- Mass Matrix M ---'); disp(M);
disp('--- Coriolis Matrix C ---'); disp(C);
disp('--- Gravity Vector G ---'); disp(G);
