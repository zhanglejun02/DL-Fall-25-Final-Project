import torch
import torch.nn.functional as F
import math

class AutoProjector:
    def __init__(
        self,
        rank=100,
        verbose=False,
        scale=1.0,
        proj_type='std',
        drift_threshold=0.5,   
        check_interval=30,
        update_condition='smaller',
        alpha=0.95,       
        ngl_eps=1e-8, 
        ngl_scale=1.01,
        use_stable=True, 
        method="traditional" 
    ):
        """
        - rank: 子空间秩
        - drift_threshold: 漂移阈值
        - check_interval: 每隔多少 step 检查漂移
        - update_condition: 'smaller' 表示 drift < threshold 就更新；'greater' 表示 drift > threshold 就更新
        """
        self.rank = rank
        self.verbose = verbose
        self.scale = scale
        self.proj_type = proj_type
        self.method = method

        self.drift_threshold = drift_threshold
        self.check_interval = check_interval
        self.update_condition = update_condition

        self.alpha = alpha
        self.ngl_eps = ngl_eps
        self.ngl_scale = ngl_scale
        self.magnitude_ema = None
        self.scaled_grad = None
        self.use_stable = use_stable

        self.ortho_matrix = None
        self.first_direction_in_subspace = None
        self.project_times = 0
        
        # 统计更新
        self.update_count = 0
        self.information = []



    def project(self, full_rank_grad):
        """
        计算流程：
          1) 做低秩投影
          2) 对投影结果low_rank_grad做“DoRA稳定化” => 得到 stable_grad
          3) 用 stable_grad 计算 direction 并跟 first_direction_in_subspace 做漂移判断
          4) 若需要更新子空间 => 重投影 + 再次 DoRA稳定化
          5) 返回最终 stable_grad
        """
        # 若 ortho_matrix 为空 => 第一次初始化
        if self.ortho_matrix is None:
            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad, self.rank, type=self._get_type(full_rank_grad)
            )
            low_rank_grad = self._project_low_rank(full_rank_grad)

            if self.use_stable:
                low_rank_grad = self._stabilize(low_rank_grad)

            direction_init = F.normalize(low_rank_grad.flatten(), dim=0, eps=1e-7)
            self.first_direction_in_subspace = direction_init.detach()
            self.project_times = 1

            return low_rank_grad
        
        low_rank_grad = self._project_low_rank(full_rank_grad)
        self.project_times += 1

        # Warm Up in new subspace
        if self.use_stable and self.project_times <= 10:
            low_rank_grad = self._stabilize(low_rank_grad)

        # 计算 low_rank_grad 的整体方向
        direction_now = F.normalize(low_rank_grad.flatten(), dim=0, eps=1e-7)
        should_update_proj = False

        # 每隔 check_interval 步检查漂移，前100步不检查
        if self.project_times % self.check_interval == 0 and self.project_times > 100:
            diff = direction_now - self.first_direction_in_subspace
            drift_val = diff.norm(p=2) / self.project_times

            if self.verbose:
                msg = f"[AutoProjector] step={self.project_times}, drift={drift_val:.6f}, threshold={self.drift_threshold}"
                self.information.append(msg)

            if self.update_condition == 'greater':
                if drift_val > self.drift_threshold:
                    should_update_proj = True
            else:  # 'smaller'
                if drift_val < self.drift_threshold:
                    should_update_proj = True

        if should_update_proj:

            self.ortho_matrix = self.get_orthogonal_matrix(
                full_rank_grad, self.rank, type=self._get_type(full_rank_grad)
            )
            # 再投影
            low_rank_grad = self._project_low_rank(full_rank_grad)

            if self.use_stable:
                _ = self._stabilize(low_rank_grad)
                low_rank_grad = self._stabilize(low_rank_grad)

            # 更新初始方向
            direction_new = F.normalize(low_rank_grad.flatten(), dim=0, eps=1e-7)
            self.first_direction_in_subspace = direction_new.detach()

            self.project_times = 1
            self.update_count += 1

        return low_rank_grad

    def _stabilize(self, grad):

        grad_norm = grad.norm(p=2) + 1e-7
        direction = grad / grad_norm

        scale_factor = math.sqrt(self.rank)
        magnitude = grad_norm / scale_factor

        if self.magnitude_ema is None:
            self.magnitude_ema = magnitude
        else:
            self.magnitude_ema = self.alpha * self.magnitude_ema + (1 - self.alpha) * magnitude
        m = self.magnitude_ema

        # limiter
        if self.scaled_grad is not None:
            limiter = max(m / (self.scaled_grad + self.ngl_eps), self.ngl_scale) / self.ngl_scale
            m = m / limiter
            self.scaled_grad = m
        else:
            self.scaled_grad = m

        return direction * m * scale_factor

    def _get_type(self, grad):
        if self.proj_type in ['std', 'reverse_std']:
            return 'right' if grad.shape[0] >= grad.shape[1] else 'left'
        return self.proj_type

    def _project_low_rank(self, full_rank_grad):
        P = self.ortho_matrix
        device = full_rank_grad.device
        dtype = full_rank_grad.dtype

        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                return torch.matmul(full_rank_grad, P.t().to(device, dtype=dtype))
            else:
                return torch.matmul(P.t().to(device, dtype=dtype), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                return torch.matmul(P.t().to(device=device, dtype=dtype), full_rank_grad)
            else:
                return torch.matmul(full_rank_grad, P.t().to(device=device, dtype=dtype))
        elif self.proj_type == 'right':
            return torch.matmul(full_rank_grad, P.t().to(device=device, dtype=dtype))
        elif self.proj_type == 'left':
            return torch.matmul(P.t().to(device=device, dtype=dtype), full_rank_grad)
        elif self.proj_type == 'full':
            return torch.matmul(P[0].t().to(device=device, dtype=dtype), full_rank_grad) @ P[1].t().to(device)
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

    def project_back(self, low_rank_grad):
        device = low_rank_grad.device
        dtype  = low_rank_grad.dtype

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(device=device, dtype=dtype))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(device=device, dtype=dtype), low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(device=device, dtype=dtype), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(device=device, dtype=dtype))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(device=device, dtype=dtype))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(device=device, dtype=dtype), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(device=device, dtype=dtype), low_rank_grad) @ self.ortho_matrix[1].to(low_rank_grad.device)
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank, type, method="traditional"):
        matrix = weights.data.float().cuda()  # 确保在GPU上运行
        n, m = matrix.shape
        k = rank

        if method == "traditional":
            # 步骤 1：生成随机矩阵 Omega，维度 (m, k)
            omega = torch.randn(m, k, device=matrix.device)

            # 步骤 2：计算 Y = A @ Omega，维度 (n, k)
            Y = torch.matmul(matrix, omega)

            # 步骤 3：对 Y 进行 QR 分解，得到 Q，维度 (n, k)
            Q, _ = torch.linalg.qr(Y, mode='reduced')

            # 步骤 4：计算 B = Q.T @ A，维度 (k, m)
            B = torch.matmul(Q.t(), matrix)

            # 步骤 5：对 B 进行 SVD 分解，得到 U_hat, S, Vh，维度分别为 (k, k), (k,), (k, m)
            U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)

            # 步骤 6：计算 U = Q @ U_hat，维度 (n, k)
            U = torch.matmul(Q, U_hat)

        else:
            if k < 128:
                U, S, Vh = torch.svd_lowrank(matrix, q=2*k, niter=2)
            else:
                U, S, Vh = torch.svd_lowrank(matrix, q=k+10, niter=3)

        # 根据类型返回相应的正交矩阵
        if type == 'right':
            return Vh[:rank, :]  # (k, m)
        elif type == 'left':
            return U[:, :rank]  # (n, k)
        elif type == 'full':
            return [U[:, :rank], Vh[:rank, :]]  # [(n, k), (k, m)]
        else:
            raise ValueError("type 应为 'left'、'right' 或 'full'")
