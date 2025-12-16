import torch

class HMTProjector:
    def __init__(self, rank=100, verbose=False, update_proj_gap=10, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device))
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device))
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t().to(full_rank_grad.device), full_rank_grad) @ self.ortho_matrix[1].t().to(full_rank_grad.device)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device))
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device), low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # 注意这里与 'std' 不同
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device), low_rank_grad) @ self.ortho_matrix[1].to(low_rank_grad.device)

        return full_rank_grad * self.scale

    # 使用 HMT 方法替代 SVD
    def get_orthogonal_matrix(self, weights, rank, type):
        matrix = weights.data.float().cuda()  # 确保在GPU上运行
        n, m = matrix.shape
        k = rank

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

        # 根据类型返回相应的正交矩阵
        if type == 'right':
            return Vh[:rank, :]  # (k, m)
        elif type == 'left':
            return U[:, :rank]  # (n, k)
        elif type == 'full':
            return [U[:, :rank], Vh[:rank, :]]  # [(n, k), (k, m)]
        else:
            raise ValueError("type 应为 'left'、'right' 或 'full'")