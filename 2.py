import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from numpy.linalg import norm, cholesky, solve
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# --- 1. 参数设置 ---
n_samples = 200
n_features = 1000
max_iter = 250
n_trials = 10
lambda_ratio = 0.1
convergence_tol = 1e-6

# 绘图配置：颜色/线型/线宽
algo_configs = {
    'Coordinate Desc': {'color': 'firebrick', 'style': '-', 'width': 2.5},
    'Coordinate Desc (Pathwise+Active)': {'color': 'darkred', 'style': '--', 'width': 2.5},
    'Huber Gradient': {'color': 'limegreen', 'style': '-', 'width': 2.5},
    'Huber Gradient (Accel)': {'color': 'forestgreen', 'style': '-', 'width': 2.5},
    'Huber (Accel + Restart)': {'color': 'darkgreen', 'style': '--', 'width': 2.5},
    'FISTA': {'color': 'darkblue', 'style': '-', 'width': 2.5},
    'FISTA (Restart)': {'color': 'blue', 'style': '--', 'width': 2.5},
    'ADMM (rho=0.5)': {'color': 'orange', 'style': '-', 'width': 2},
    'ADMM (rho=1)': {'color': 'magenta', 'style': '-', 'width': 2},
    'ADMM (rho=2)': {'color': 'cyan', 'style': '-', 'width': 2},
    'ADMM (rho=5)': {'color': 'purple', 'style': '-', 'width': 2},
    'Proximal Gradient (ISTA)': {'color': 'brown', 'style': '-', 'width': 2},
    'Subgradient': {'color': 'gray', 'style': ':', 'width': 2},
    'Continuation Subgradient': {'color': 'teal', 'style': '-', 'width': 2},
    'Stochastic Subgradient': {'color': 'darkcyan', 'style': ':', 'width': 2},
    'Stochastic Proximal Gradient': {'color': 'olive', 'style': '--', 'width': 2},
    'Primal-Dual Hybrid Gradient': {'color': 'black', 'style': '-', 'width': 2},
}

results = {name: [] for name in algo_configs}


# --- 2. 辅助函数 ---
def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)


def lasso_objective(beta, X, y, n, lam):
    residual = X @ beta - y
    l2_loss = (0.5 / n) * (residual @ residual)
    l1_norm = lam * norm(beta, 1)
    return l2_loss + l1_norm


def iterations_to_converge(history, tol):
    for idx, value in enumerate(history):
        if value <= tol:
            return idx + 1
    return len(history)


def l1_subgradient(beta, grad_f, lam):
    s = np.sign(beta)
    zero_idx = np.where(np.abs(beta) <= 1e-12)[0]
    if zero_idx.size > 0:
        s_zero = -grad_f[zero_idx] / (lam + 1e-12)
        s_zero = np.clip(s_zero, -1.0, 1.0)
        s[zero_idx] = s_zero
    return grad_f + lam * s


# --- 3. 算法实现 ---
def coordinate_descent(X, y, n, p, lam, max_iter, f_star):
    beta = np.zeros(p)
    history = []
    A_j = np.maximum(np.sum(X ** 2, axis=0) / n, 1e-8)

    for k in range(max_iter):
        for j in range(p):
            old_beta_j = beta[j]
            residual_no_j = y - (X @ beta - X[:, j] * old_beta_j)
            c_j = (X[:, j] @ residual_no_j) / n
            beta[j] = soft_threshold(c_j / A_j[j], lam / A_j[j])
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def coordinate_descent_pathwise_active(X, y, n, p, lam_target, max_iter, f_star,
                                      n_lambdas=10, active_check_every=5):
    lam_max = norm(X.T @ y, ord=np.inf) / n
    if lam_target >= lam_max:
        return [max(lasso_objective(np.zeros(p), X, y, n, lam_target) - f_star, 1e-15)] * max_iter

    lambdas = np.geomspace(lam_max, lam_target, num=n_lambdas)
    beta = np.zeros(p)
    history = []
    A_j = np.maximum(np.sum(X ** 2, axis=0) / n, 1e-8)

    for lam in lambdas:
        active = set(np.where(np.abs(beta) > 1e-12)[0].tolist())
        for j in range(p):
            old = beta[j]
            residual_no_j = y - (X @ beta - X[:, j] * old)
            c_j = (X[:, j] @ residual_no_j) / n
            new = soft_threshold(c_j / A_j[j], lam / A_j[j])
            if np.abs(new) > 0:
                active.add(j)
            beta[j] = new

        iter_per_lambda = max(5, max_iter // max(1, len(lambdas)))
        for it in range(iter_per_lambda):
            if len(active) == 0:
                for j in range(p):
                    old = beta[j]
                    residual_no_j = y - (X @ beta - X[:, j] * old)
                    c_j = (X[:, j] @ residual_no_j) / n
                    new = soft_threshold(c_j / A_j[j], lam / A_j[j])
                    if np.abs(new) > 0:
                        active.add(j)
                    beta[j] = new
            else:
                for j in list(active):
                    old = beta[j]
                    residual_no_j = y - (X @ beta - X[:, j] * old)
                    c_j = (X[:, j] @ residual_no_j) / n
                    beta[j] = soft_threshold(c_j / A_j[j], lam / A_j[j])

            if (it + 1) % active_check_every == 0:
                residual = X @ beta - y
                grad = (X.T @ residual) / n
                for j in range(p):
                    if j in active:
                        continue
                    if np.abs(grad[j]) > lam * 0.999:
                        active.add(j)

    beta_start = beta.copy()
    beta = beta_start.copy()
    A_j = np.maximum(np.sum(X ** 2, axis=0) / n, 1e-8)

    for k in range(max_iter):
        active = set(np.where(np.abs(beta) > 1e-12)[0].tolist())
        if len(active) == 0:
            for j in range(p):
                old_beta_j = beta[j]
                residual_no_j = y - (X @ beta - X[:, j] * old_beta_j)
                c_j = (X[:, j] @ residual_no_j) / n
                beta[j] = soft_threshold(c_j / A_j[j], lam_target / A_j[j])
        else:
            for j in list(active):
                old_beta_j = beta[j]
                residual_no_j = y - (X @ beta - X[:, j] * old_beta_j)
                c_j = (X[:, j] @ residual_no_j) / n
                beta[j] = soft_threshold(c_j / A_j[j], lam_target / A_j[j])
            if (k + 1) % 5 == 0:
                residual = X @ beta - y
                grad = (X.T @ residual) / n
                for j in range(p):
                    if np.abs(beta[j]) <= 1e-12 and np.abs(grad[j]) > lam_target * 0.999:
                        old_beta_j = beta[j]
                        residual_no_j = y - (X @ beta - X[:, j] * old_beta_j)
                        c_j = (X[:, j] @ residual_no_j) / n
                        beta[j] = soft_threshold(c_j / A_j[j], lam_target / A_j[j])

        subopt = lasso_objective(beta, X, y, n, lam_target) - f_star
        history.append(max(subopt, 1e-15))

    return history


def gradient_descent_huber(X, y, n, p, lam, max_iter, f_star, delta=1e-2):
    beta = np.zeros(p)
    history = []
    L_f = norm(X.T @ X / n, ord=2)
    L_g = lam / delta
    alpha = 1.0 / (L_f + L_g + 1e-12)
    for k in range(max_iter):
        grad_f = (X.T @ (X @ beta - y)) / n
        grad_g_huber = lam * np.clip(beta / delta, -1, 1)
        beta = beta - alpha * (grad_f + grad_g_huber)
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def gradient_descent_huber_accel(X, y, n, p, lam, max_iter, f_star, delta=1e-2):
    beta = np.zeros(p)
    z = np.zeros(p)
    t = 1.0
    history = []
    L_f = norm(X.T @ X / n, ord=2)
    L_g = lam / delta
    alpha = 1.0 / (L_f + L_g + 1e-12)
    for k in range(max_iter):
        beta_old = beta.copy()
        grad_f = (X.T @ (X @ z - y)) / n
        grad_g_huber = lam * np.clip(z / delta, -1, 1)
        beta = z - alpha * (grad_f + grad_g_huber)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = beta + ((t - 1) / t_new) * (beta - beta_old)
        t = t_new
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def gradient_descent_huber_accel_restart(X, y, n, p, lam, max_iter, f_star, delta=1e-2):
    beta = np.zeros(p)
    z = np.zeros(p)
    t = 1.0
    history = []
    L_f = norm(X.T @ X / n, ord=2)
    L_g = lam / delta
    alpha = 1.0 / (L_f + L_g + 1e-12)

    for k in range(max_iter):
        beta_old = beta.copy()
        grad_f = (X.T @ (X @ z - y)) / n
        grad_g_huber = lam * np.clip(z / delta, -1, 1)
        grad = grad_f + grad_g_huber
        beta = z - alpha * grad

        if np.dot(beta - beta_old, beta - z) < 0:
            t_new = 1.0
            z = beta
        else:
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = beta + ((t - 1) / t_new) * (beta - beta_old)
        t = t_new
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def fista(X, y, n, lam, max_iter, f_star):
    beta = np.zeros(n_features)
    z = np.zeros(n_features)
    t = 1.0
    history = []
    L = norm(X.T @ X / n, ord=2) + 1e-12
    alpha = 1.0 / L
    for k in range(max_iter):
        beta_old = beta.copy()
        grad_z = (X.T @ (X @ z - y)) / n
        beta = soft_threshold(z - alpha * grad_z, alpha * lam)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z = beta + ((t - 1) / t_new) * (beta - beta_old)
        t = t_new
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def fista_restart(X, y, n, lam, max_iter, f_star):
    beta = np.zeros(n_features)
    z = np.zeros(n_features)
    t = 1.0
    history = []
    L = norm(X.T @ X / n, ord=2) + 1e-12
    alpha = 1.0 / L

    for k in range(max_iter):
        beta_old = beta.copy()
        grad_z = (X.T @ (X @ z - y)) / n
        beta_new = soft_threshold(z - alpha * grad_z, alpha * lam)

        if np.dot(z - beta_new, beta_new - beta_old) > 0:
            t_new = 1.0
            z = beta_new
        else:
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = beta_new + ((t - 1) / t_new) * (beta_new - beta_old)

        beta = beta_new
        t = t_new
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def admm(X, y, n, p, lam, rho, max_iter, f_star):
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    history = []
    I = np.identity(p)
    M = X.T @ X / n + rho * I + 1e-12 * I
    try:
        L_cho = cholesky(M)
    except Exception:
        M += 1e-6 * I
        L_cho = cholesky(M)

    for k in range(max_iter):
        rhs = (X.T @ y / n) + rho * (z - u)
        beta = solve(L_cho.T, solve(L_cho, rhs))
        z = soft_threshold(beta + u, lam / rho)
        u = u + beta - z
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def proximal_gradient_ista(X, y, n, p, lam, max_iter, f_star):
    beta = np.zeros(p)
    history = []
    L = norm(X.T @ X / n, ord=2) + 1e-12
    alpha = 1.0 / L
    for k in range(max_iter):
        grad = (X.T @ (X @ beta - y)) / n
        beta = soft_threshold(beta - alpha * grad, alpha * lam)
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def subgradient_method(X, y, n, p, lam, max_iter, f_star, step0=None):
    beta = np.zeros(p)
    history = []
    L = norm(X.T @ X / n, ord=2) + 1e-12
    if step0 is None:
        step0 = 1.0 / (L + lam)
    for k in range(max_iter):
        residual = X @ beta - y
        grad_f = (X.T @ residual) / n
        g = l1_subgradient(beta, grad_f, lam)
        step = step0 / np.sqrt(k + 1.0)
        beta = beta - step * g
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def continuation_subgradient(X, y, n, p, lam_target, max_iter, f_star, stages=5):
    lam_max = norm(X.T @ y, ord=np.inf) / n
    if lam_max <= lam_target:
        return subgradient_method(X, y, n, p, lam_target, max_iter, f_star)
    lam_schedule = np.geomspace(lam_max, lam_target, num=stages)
    beta = np.zeros(p)
    history = []
    L = norm(X.T @ X / n, ord=2) + 1e-12
    step0 = 1.0 / (L + lam_schedule[0])
    base_iters = max(1, max_iter // stages)

    for idx, lam_stage in enumerate(lam_schedule):
        if len(history) >= max_iter:
            break
        stage_iters = base_iters if idx < stages - 1 else max_iter - len(history)
        for _ in range(stage_iters):
            grad_f = (X.T @ (X @ beta - y)) / n
            g = l1_subgradient(beta, grad_f, lam_stage)
            step = step0 / np.sqrt(len(history) + 1.0)
            beta = beta - step * g
            subopt = lasso_objective(beta, X, y, n, lam_target) - f_star
            history.append(max(subopt, 1e-15))
            if len(history) >= max_iter:
                break

    if len(history) < max_iter:
        history.extend([history[-1]] * (max_iter - len(history)))
    return history[:max_iter]


def stochastic_subgradient_method(X, y, n, p, lam, max_iter, f_star, batch_size=32):
    beta = np.zeros(p)
    history = []
    batch = min(batch_size, n)
    rng = np.random.default_rng()
    L = norm(X, ord=2) ** 2 / n + 1e-12
    step0 = 1.0 / (L + lam)

    for k in range(max_iter):
        idx = rng.choice(n, size=batch, replace=False)
        Xb = X[idx]
        yb = y[idx]
        residual = Xb @ beta - yb
        grad_f = (Xb.T @ residual) / batch
        g = l1_subgradient(beta, grad_f, lam)
        step = step0 / np.sqrt(k + 1.0)
        beta = beta - step * g
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def stochastic_proximal_gradient(X, y, n, p, lam, max_iter, f_star, batch_size=32):
    beta = np.zeros(p)
    history = []
    rng = np.random.default_rng()
    batch = min(batch_size, n)
    L = norm(X, ord=2) ** 2 / n + 1e-12
    alpha = 0.5 / L

    for k in range(max_iter):
        idx = rng.choice(n, size=batch, replace=False)
        Xb = X[idx]
        yb = y[idx]
        grad = (Xb.T @ (Xb @ beta - yb)) / batch
        beta = soft_threshold(beta - alpha * grad, alpha * lam)
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


def primal_dual_hybrid_gradient(X, y, n, p, lam, max_iter, f_star):
    beta = np.zeros(p)
    beta_bar = np.zeros(p)
    dual = np.zeros(n)
    history = []
    K_norm = norm(X, ord=2) + 1e-12
    tau = 0.9 / (K_norm ** 2)
    sigma = 0.9 / (K_norm ** 2)
    theta = 1.0

    for k in range(max_iter):
        dual = dual + sigma * (X @ beta_bar)
        dual = (dual - sigma * y) / (1.0 + sigma * n)
        grad_term = X.T @ dual
        beta_next = soft_threshold(beta - tau * grad_term, tau * lam)
        beta_bar = beta_next + theta * (beta_next - beta)
        beta = beta_next
        subopt = lasso_objective(beta, X, y, n, lam) - f_star
        history.append(max(subopt, 1e-15))
    return history


# --- 4. 主循环 ---
print(f"Starting {n_trials} random trials...")

for i in range(n_trials):
    if (i + 1) % 5 == 0:
        print(f"Processing trial {i + 1}/{n_trials}...")

    X = np.random.randn(n_samples, n_features)
    true_beta = np.zeros(n_features)
    true_beta[:10] = np.random.uniform(-5, 5, 10)
    y = X @ true_beta + np.random.randn(n_samples) * 0.5

    lam_max = norm(X.T @ y, ord=np.inf) / n_samples
    lam = lam_max * lambda_ratio

    lasso_sklearn = Lasso(alpha=lam, fit_intercept=False, tol=1e-14, max_iter=20000)
    lasso_sklearn.fit(X, y)
    f_star = lasso_objective(lasso_sklearn.coef_, X, y, n_samples, lam)

    results['Coordinate Desc'].append(coordinate_descent(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Coordinate Desc (Pathwise+Active)'].append(
        coordinate_descent_pathwise_active(X, y, n_samples, n_features, lam, max_iter, f_star)
    )
    results['Huber Gradient'].append(gradient_descent_huber(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Huber Gradient (Accel)'].append(
        gradient_descent_huber_accel(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Huber (Accel + Restart)'].append(
        gradient_descent_huber_accel_restart(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['FISTA'].append(fista(X, y, n_samples, lam, max_iter, f_star))
    results['FISTA (Restart)'].append(fista_restart(X, y, n_samples, lam, max_iter, f_star))
    results['ADMM (rho=0.5)'].append(admm(X, y, n_samples, n_features, lam, 0.5, max_iter, f_star))
    results['ADMM (rho=1)'].append(admm(X, y, n_samples, n_features, lam, 1.0, max_iter, f_star))
    results['ADMM (rho=2)'].append(admm(X, y, n_samples, n_features, lam, 2.0, max_iter, f_star))
    results['ADMM (rho=5)'].append(admm(X, y, n_samples, n_features, lam, 5.0, max_iter, f_star))
    results['Proximal Gradient (ISTA)'].append(
        proximal_gradient_ista(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Subgradient'].append(subgradient_method(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Continuation Subgradient'].append(
        continuation_subgradient(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Stochastic Subgradient'].append(
        stochastic_subgradient_method(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Stochastic Proximal Gradient'].append(
        stochastic_proximal_gradient(X, y, n_samples, n_features, lam, max_iter, f_star))
    results['Primal-Dual Hybrid Gradient'].append(
        primal_dual_hybrid_gradient(X, y, n_samples, n_features, lam, max_iter, f_star))

# --- 5. 收敛统计 ---
convergence_summary = {}
for name, histories in results.items():
    if not histories:
        continue
    iter_counts = [iterations_to_converge(history, convergence_tol) for history in histories]
    convergence_summary[name] = np.array(iter_counts)

print(f"\n各算法在 tol={convergence_tol:.0e} 下的收敛迭代次数统计：")
for name, counts in convergence_summary.items():
    avg_iters = counts.mean()
    median_iters = np.median(counts)
    min_iters = counts.min()
    max_iters = counts.max()
    conv_ratio = np.mean(counts < max_iter) * 100
    print(f"{name:35s}: 平均 {avg_iters:6.2f} 次 中位 {median_iters:6.2f}, "
          f"最少 {int(min_iters):3d}, 最多 {int(max_iters):3d}, 提前收敛比例 {conv_ratio:5.1f}%")

print("All trials complete. Plotting...")

# --- 6. 绘图 ---
plt.figure(figsize=(10, 8))
k_axis = np.arange(1, max_iter + 1)

for name, histories in results.items():
    data_matrix = np.array(histories)
    min_len = min(len(h) for h in histories)
    data_matrix = data_matrix[:, :min_len]
    current_k_axis = k_axis[:min_len]
    mean_curve = np.mean(data_matrix, axis=0)

    cfg = algo_configs[name]
    color = cfg['color']
    style = cfg['style']
    width = cfg.get('width', 2)

    for single_run in data_matrix:
        plt.plot(current_k_axis, single_run, color=color, alpha=0.08, linewidth=0.5)

    plt.plot(current_k_axis, mean_curve, color=color, linestyle=style, linewidth=width, label=name)

plt.yscale('log')
plt.xlabel('Iteration k', fontsize=14)
plt.ylabel('Suboptimality $f(x_k) - f^*$', fontsize=14)
plt.title('LASSO Convergence: Algorithms with Dynamic Strategies', fontsize=16)
plt.legend(fontsize=9, loc='upper right', framealpha=0.9, ncol=2)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.ylim(bottom=1e-10)
plt.xlim(0, max_iter)
plt.tight_layout()
plt.show()
