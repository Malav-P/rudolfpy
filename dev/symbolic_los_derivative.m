% symbolic calculatino of los

clear; close all; clc;

syms x y z

r = [x;y;z];
los = r/norm(r);
symJ = jacobian(los, r);
func_jac = matlabFunction(symJ);

rvec = [1.0, 2.0, 3.0]';
ans1 = func_jac(rvec(1),rvec(2),rvec(3))
ans2 = get_jac(rvec)

ans2 - ans1



function H = get_jac(rvec)
H = eye(3)/norm(rvec) - rvec*transpose(rvec)/norm(rvec)^3;
end