clc; clear;

%% ================= SYMBOLIC VARIABLES =================
syms th1 th2 th3 th4 real
syms dth1 dth2 dth3 dth4 real
syms l1 l2 l3 l4 r1 r2 r3 r4 m1 m2 m3 m4 g real

q  = [th1; th2; th3; th4];
dq = [dth1; dth2; dth3; dth4];

%% ================= CENTER OF MASS POSITIONS =================
c2 = l2*cos(th2)/2;
c3 = l2*cos(th2) + l3*cos(th2+th3)/2;
c4 = l2*cos(th2) + l3*cos(th2+th3) + l4*cos(th2+th3+th4)/2;

s2 = l2*sin(th2)/2;
s3 = l2*sin(th2) + l3*sin(th2+th3)/2;
s4 = l2*sin(th2) + l3*sin(th2+th3) + l4*sin(th2+th3+th4)/2;

p1 = [0; 0; l1/2];
p2 = [ c2*cos(th1); c2*sin(th1); (l1 - s2) ];
p3 = [ c3*cos(th1); c3*sin(th1); (l1 - s3) ];
p4 = [ c4*cos(th1); c4*sin(th1); (l1 - s4) ];

%% ================ VELOCITIES =================
v1 = jacobian(p1,q)*dq;
v2 = jacobian(p2,q)*dq;
v3 = jacobian(p3,q)*dq;
v4 = jacobian(p4,q)*dq;

%% ================= INERTIAS =================
I1 = (1/2) * m1 * (r1^2);
I2 = (1/12)* m2 * (3*r2^2 + l2^2);
I3 = (1/12)* m3 * (3*r3^2 + l3^2);
I4 = (1/12)* m4 * (3*r4^2 + l4^2);

%% ================= KINETIC ENERGY =================
T =                    0.5*I1*dth1^2 + ...
    0.5*m2*(v2.'*v2) + 0.5*I2*dth2^2 + ...
    0.5*m3*(v3.'*v3) + 0.5*I3*(dth2 + dth3)^2 + ...
    0.5*m4*(v4.'*v4) + 0.5*I4*(dth2 + dth3 + dth4)^2;
T = simplify(T);

%% ================= POTENTIAL ENERGY =================
V = g * (m1*p1(3) + ...
         m2*p2(3) + ...
         m3*p3(3) + ...
         m4*p4(3));
V = simplify(V);

%% ================= MASS MATRIX =================
M = sym(zeros(4));
for i = 1:4
    for j = 1:4
        M(i,j) = diff(diff(T,dq(i)),dq(j));
    end
end
M = simplify(M);

%% ================= CORIOLIS MATRIX =================
C = sym(zeros(4));
for i = 1:4
    for j = 1:4
        for k = 1:4
            C(i,j) = C(i,j) + 1/2 * ...
                ( diff(M(i,j),q(k)) ...
                + diff(M(i,k),q(j)) ...
                - diff(M(j,k),q(i)) ) * dq(k);
        end
        C(i,j) = simplify(C(i,j));
    end
end

%% ================= GRAVITY VECTOR =================
G = jacobian(V,q).';

%% ================= DISPLAY =================
disp('--- Mass Matrix M(q) ---'); disp(M);
disp('--- Coriolis Matrix C(q,dq) ---'); disp(C);
disp('--- Gravity Vector G(q) ---'); disp(G);
