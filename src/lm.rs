//! Port of scipy.optimize.least_squares (trf, no bounds, x_scale="jac") plus small dense linear algebra.

pub struct LmResult {
    pub x: Vec<f64>,
    pub fun: Vec<f64>,
    /// Jacobian at `x`, row-major `fun.len() x x.len()`.
    pub jac: Vec<Vec<f64>>,
    pub nfev: usize,
    pub success: bool,
}

impl LmResult {
    pub fn cost(&self) -> f64 {
        0.5 * self.fun.iter().map(|r| r * r).sum::<f64>()
    }
    pub fn rms(&self) -> f64 {
        (self.fun.iter().map(|r| r * r).sum::<f64>() / self.fun.len().max(1) as f64).sqrt()
    }
    /// pinv(J^T J) * s2.
    pub fn covariance(&self, s2: f64) -> Vec<Vec<f64>> {
        let m = self.x.len();
        let mut a = vec![vec![0.0; m]; m];
        for row in &self.jac {
            for i in 0..m {
                for j in 0..m {
                    a[i][j] += row[i] * row[j];
                }
            }
        }
        pinv_sym(&a).into_iter().map(|r| r.into_iter().map(|v| v * s2).collect()).collect()
    }
    /// Standard errors with s^2 = mean squared residual (as rms^2 in the Python kit).
    pub fn errors(&self, extra_scale: f64) -> Vec<f64> {
        let c = self.covariance(self.rms().powi(2) * extra_scale);
        (0..self.x.len()).map(|i| c[i][i].max(0.0).sqrt()).collect()
    }
}

/// numpy's pairwise summation (8-way unrolled blocks of up to 128), so float32 sums round like ndarray.sum().
pub fn np_sum<T: Copy + Default + std::ops::Add<Output = T>>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 {
        a.iter().fold(T::default(), |s, &v| s + v)
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let full = n - n % 8;
        for c in a[8..full].chunks_exact(8) {
            for (rk, &v) in r.iter_mut().zip(c) {
                *rk = *rk + v;
            }
        }
        a[full..].iter().fold(((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7])), |s, &v| s + v)
    } else {
        let n2 = n / 2 - (n / 2) % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// scipy approx_derivative '2-point': forward step sqrt(eps) * sign(x) * max(1, |x|).
pub fn jac_2point<F: Fn(&[f64]) -> Vec<f64>>(f: &F, x: &[f64], f0: &[f64]) -> Vec<Vec<f64>> {
    let mut j = vec![vec![0.0; x.len()]; f0.len()];
    for k in 0..x.len() {
        let h = f64::EPSILON.sqrt() * if x[k] >= 0.0 { 1.0 } else { -1.0 } * x[k].abs().max(1.0);
        let mut xp = x.to_vec();
        xp[k] += h;
        let fp = f(&xp);
        let dx = xp[k] - x[k];
        for i in 0..f0.len() {
            j[i][k] = (fp[i] - f0[i]) / dx;
        }
    }
    j
}

/// Thin SVD of a tall matrix by one-sided Jacobi: (U columns, s descending, V columns).
fn svd(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = a.first().map_or(0, |r| r.len());
    let mut u: Vec<Vec<f64>> = (0..n).map(|k| a.iter().map(|r| r[k]).collect()).collect();
    let mut v: Vec<Vec<f64>> = (0..n).map(|k| (0..n).map(|i| if i == k { 1.0 } else { 0.0 }).collect()).collect();
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (al, be) = (u[p].iter().map(|x| x * x).sum::<f64>(), u[q].iter().map(|x| x * x).sum::<f64>());
                let ga: f64 = u[p].iter().zip(&u[q]).map(|(x, y)| x * y).sum();
                if ga == 0.0 || ga.abs() <= 1e-15 * (al * be).sqrt() {
                    continue;
                }
                rotated = true;
                let ze = (be - al) / (2.0 * ga);
                let t = ze.signum() / (ze.abs() + (1.0 + ze * ze).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = c * t;
                for w in [&mut u, &mut v] {
                    let (lo, hi) = w.split_at_mut(q);
                    for (x, y) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
                        let (xp, yq) = (*x, *y);
                        *x = c * xp - sn * yq;
                        *y = sn * xp + c * yq;
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let s: Vec<f64> = u.iter().map(|c| norm(c)).collect();
    let mut o: Vec<usize> = (0..n).collect();
    o.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    let uc = o.iter().map(|&k| u[k].iter().map(|x| if s[k] > 0.0 { x / s[k] } else { 0.0 }).collect()).collect();
    (uc, o.iter().map(|&k| s[k]).collect(), o.iter().map(|&k| v[k].clone()).collect())
}

/// scipy common.solve_lsq_trust_region (exact solver); v holds V's columns.
fn solve_lsq_trust_region(n: usize, m: usize, uf: &[f64], s: &[f64], v: &[Vec<f64>], delta: f64, initial_alpha: f64) -> (Vec<f64>, f64) {
    let suf: Vec<f64> = s.iter().zip(uf).map(|(a, b)| a * b).collect();
    let vdot = |c: &[f64]| -> Vec<f64> { (0..n).map(|i| -(0..n).map(|k| v[k][i] * c[k]).sum::<f64>()).collect() };
    let phi_d = |alpha: f64| {
        let den: Vec<f64> = s.iter().map(|x| x * x + alpha).collect();
        let pn = norm(&suf.iter().zip(&den).map(|(a, d)| a / d).collect::<Vec<_>>());
        (pn - delta, -suf.iter().zip(&den).map(|(a, d)| a * a / (d * d * d)).sum::<f64>() / pn)
    };
    let full_rank = m >= n && s[n - 1] > f64::EPSILON * m as f64 * s[0];
    if full_rank {
        let p = vdot(&uf.iter().zip(s).map(|(a, b)| a / b).collect::<Vec<_>>());
        if norm(&p) <= delta {
            return (p, 0.0);
        }
    }
    let mut upper = norm(&suf) / delta;
    let mut lower = if full_rank { let (ph, pp) = phi_d(0.0); -ph / pp } else { 0.0 };
    let mut alpha = if !full_rank && initial_alpha == 0.0 { (0.001 * upper).max((lower * upper).sqrt()) } else { initial_alpha };
    for _ in 0..10 {
        if alpha < lower || alpha > upper {
            alpha = (0.001 * upper).max((lower * upper).sqrt());
        }
        let (phi, pp) = phi_d(alpha);
        if phi < 0.0 {
            upper = alpha;
        }
        let ratio = phi / pp;
        lower = lower.max(alpha - ratio);
        alpha -= (phi + delta) * ratio / delta;
        if phi.abs() < 0.01 * delta {
            break;
        }
    }
    let mut p = vdot(&suf.iter().zip(s).map(|(a, b)| a / (b * b + alpha)).collect::<Vec<_>>());
    let k = delta / norm(&p);
    p.iter_mut().for_each(|x| *x *= k);
    (p, alpha)
}

/// scipy.optimize.least_squares(fun, x0, x_scale="jac") with the defaults: trf without bounds,
/// exact trust-region solver, '2-point' Jacobian, ftol = xtol = gtol = 1e-8, max_nfev = 100 n.
pub fn solve<F: Fn(&[f64]) -> Vec<f64>>(f: F, x0: &[f64]) -> LmResult {
    let tol = 1e-8;
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut r = f(&x);
    let m = r.len();
    let mut nfev = 1;
    let mut jm = jac_2point(&f, &x, &r);
    let mut cost = 0.5 * r.iter().map(|v| v * v).sum::<f64>();
    let grad = |j: &[Vec<f64>], r: &[f64]| -> Vec<f64> { (0..n).map(|k| j.iter().zip(r).map(|(row, ri)| row[k] * ri).sum()).collect() };
    let colnorm = |j: &[Vec<f64>]| -> Vec<f64> { (0..n).map(|k| j.iter().map(|row| row[k] * row[k]).sum::<f64>().sqrt()).collect() };
    let mut g = grad(&jm, &r);
    let mut scale_inv: Vec<f64> = colnorm(&jm).into_iter().map(|c| if c == 0.0 { 1.0 } else { c }).collect();
    let mut delta = norm(&x.iter().zip(&scale_inv).map(|(a, b)| a * b).collect::<Vec<_>>());
    if delta == 0.0 {
        delta = 1.0;
    }
    let max_nfev = 100 * n;
    let mut alpha = 0.0;
    let mut status: Option<u8> = None;
    loop {
        if g.iter().fold(0.0f64, |a, v| a.max(v.abs())) < tol {
            status = Some(1);
        }
        if status.is_some() || nfev == max_nfev {
            break;
        }
        let d: Vec<f64> = scale_inv.iter().map(|v| 1.0 / v).collect();
        let g_h: Vec<f64> = g.iter().zip(&d).map(|(a, b)| a * b).collect();
        let j_h: Vec<Vec<f64>> = jm.iter().map(|row| row.iter().zip(&d).map(|(a, b)| a * b).collect()).collect();
        let (u, s, v) = svd(&j_h);
        let uf: Vec<f64> = u.iter().map(|c| c.iter().zip(&r).map(|(a, b)| a * b).sum()).collect();
        let mut actual = -1.0;
        let (mut xn, mut rn, mut cn) = (x.clone(), r.clone(), cost);
        while actual <= 0.0 && nfev < max_nfev {
            let (step_h, a2) = solve_lsq_trust_region(n, m, &uf, &s, &v, delta, alpha);
            alpha = a2;
            let js: Vec<f64> = j_h.iter().map(|row| row.iter().zip(&step_h).map(|(a, b)| a * b).sum()).collect();
            let pred = -(0.5 * js.iter().map(|v| v * v).sum::<f64>() + step_h.iter().zip(&g_h).map(|(a, b)| a * b).sum::<f64>());
            let step: Vec<f64> = step_h.iter().zip(&d).map(|(a, b)| a * b).collect();
            xn = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            rn = f(&xn);
            nfev += 1;
            let shn = norm(&step_h);
            if !rn.iter().all(|v| v.is_finite()) {
                delta = 0.25 * shn;
                continue;
            }
            cn = 0.5 * rn.iter().map(|v| v * v).sum::<f64>();
            actual = cost - cn;
            let ratio = if pred > 0.0 { actual / pred } else if pred == 0.0 && actual == 0.0 { 1.0 } else { 0.0 };
            let delta_new = if ratio < 0.25 { 0.25 * shn } else if ratio > 0.75 && shn > 0.95 * delta { 2.0 * delta } else { delta };
            let (ft, xt) = (actual < tol * cost && ratio > 0.25, norm(&step) < tol * (tol + norm(&x)));
            if ft || xt {
                status = Some(2);
                break;
            }
            alpha *= delta / delta_new;
            delta = delta_new;
        }
        if actual > 0.0 {
            x = xn;
            r = rn;
            cost = cn;
            jm = jac_2point(&f, &x, &r);
            g = grad(&jm, &r);
            scale_inv = colnorm(&jm).iter().zip(&scale_inv).map(|(a, b)| a.max(*b)).collect();
        }
    }
    LmResult { x, fun: r, jac: jm, nfev, success: status.is_some() }
}

/// Householder QR least squares for an overdetermined system; None when rank deficient.
pub fn lstsq(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let (n, m) = (a.len(), a.first().map_or(0, |r| r.len()));
    if n < m {
        return None;
    }
    for k in 0..m {
        let norm = (k..n).map(|i| a[i][k] * a[i][k]).sum::<f64>().sqrt();
        if norm == 0.0 {
            return None;
        }
        let alpha = if a[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k..n).map(|i| a[i][k]).collect();
        v[0] -= alpha;
        let vn = v.iter().map(|x| x * x).sum::<f64>();
        if vn == 0.0 {
            continue;
        }
        for j in k..m {
            let s = 2.0 * (k..n).map(|i| v[i - k] * a[i][j]).sum::<f64>() / vn;
            for i in k..n {
                a[i][j] -= s * v[i - k];
            }
        }
        let s = 2.0 * (k..n).map(|i| v[i - k] * b[i]).sum::<f64>() / vn;
        for i in k..n {
            b[i] -= s * v[i - k];
        }
    }
    let dmax = (0..m).map(|k| a[k][k].abs()).fold(0.0, f64::max);
    let mut x = vec![0.0; m];
    for k in (0..m).rev() {
        if a[k][k].abs() <= dmax * 1e-14 {
            return None;
        }
        x[k] = (b[k] - (k + 1..m).map(|j| a[k][j] * x[j]).sum::<f64>()) / a[k][k];
    }
    Some(x)
}

/// Moore-Penrose pseudo-inverse of a symmetric matrix (Jacobi eigen, numpy rcond 1e-15).
pub fn pinv_sym(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut d: Vec<Vec<f64>> = a.to_vec();
    let mut v = vec![vec![0.0; m]; m];
    for i in 0..m {
        v[i][i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..m).flat_map(|i| (0..m).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| d[i][j] * d[i][j]).sum();
        let diag: f64 = (0..m).map(|i| d[i][i] * d[i][i]).sum();
        if off <= diag * 1e-32 || off == 0.0 {
            break;
        }
        for p in 0..m {
            for q in p + 1..m {
                if d[p][q] == 0.0 {
                    continue;
                }
                let theta = (d[q][q] - d[p][p]) / (2.0 * d[p][q]);
                let t = (if theta >= 0.0 { 1.0 } else { -1.0 }) / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..m {
                    let (dkp, dkq) = (d[k][p], d[k][q]);
                    d[k][p] = c * dkp - s * dkq;
                    d[k][q] = s * dkp + c * dkq;
                }
                for k in 0..m {
                    let (dpk, dqk) = (d[p][k], d[q][k]);
                    d[p][k] = c * dpk - s * dqk;
                    d[q][k] = s * dpk + c * dqk;
                }
                for k in 0..m {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let ev: Vec<f64> = (0..m).map(|i| d[i][i]).collect();
    let cut = 1e-15 * ev.iter().fold(0.0f64, |a, e| a.max(e.abs()));
    let mut out = vec![vec![0.0; m]; m];
    for k in 0..m {
        if ev[k].abs() <= cut {
            continue;
        }
        for i in 0..m {
            for j in 0..m {
                out[i][j] += v[i][k] * v[j][k] / ev[k];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinv_inverts() {
        let a = vec![vec![4.0, 1.0, 0.5], vec![1.0, 3.0, 0.2], vec![0.5, 0.2, 2.0]];
        let p = pinv_sym(&a);
        for i in 0..3 {
            for j in 0..3 {
                let s: f64 = (0..3).map(|k| a[i][k] * p[k][j]).sum();
                assert!((s - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exp_fit() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = t.iter().map(|t| 3.0 * (-0.7 * t).exp() + 0.5).collect();
        let r = solve(|p: &[f64]| t.iter().zip(&y).map(|(t, y)| p[0] * (-p[1] * t).exp() + p[2] - y).collect(), &[1.0, 0.1, 0.0]);
        assert!((r.x[0] - 3.0).abs() < 1e-6 && (r.x[1] - 0.7).abs() < 1e-6 && (r.x[2] - 0.5).abs() < 1e-6, "{:?}", r.x);
    }
}
