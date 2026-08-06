"""
WDCNN 스타일 1D-CNN 이진 분류기.

- 1층: 넓은 커널 (잡음 억제 + 중·저주파 패턴)
- 이후: 작은 커널 깊은 층
- 입력: (N, C, L) 원시 진동, 기본 C=2 (h/v), L=2560

Validation 사용 방식은 SVM/RF와 동일:
- 6:2:2: Train만으로 학습 → Val F1로 하이퍼파라미터 선택 → Train으로 재학습 → Val/Test 평가
- 7:3: FIXED 파라미터로 Train만 학습 → Test 평가
"""

import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('..')
from config import CNN1D_BASE_PARAMS, CNN1D_PARAM_GRID, CNN1D_FIXED_PARAMS, RANDOM_STATE
from utils.evaluation import print_evaluation_report


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CNN1DNet(nn.Module):
    """Wide-first-kernel 1D CNN (WDCNN 간소화 버전)."""

    def __init__(self, in_channels=2, channels=(16, 32, 64, 64),
                 wide_kernel=64, dropout=0.3):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.features = nn.Sequential(
            # Wide first layer
            nn.Conv1d(in_channels, c1, kernel_size=wide_kernel, stride=8, padding=wide_kernel // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Small kernels
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, 1),  # binary logit
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(-1)


class CNN1DClassifier:
    """SVM/RF와 동일한 train / predict / evaluate 인터페이스."""

    def __init__(self, params=None):
        # RF/SVM: params or BASE/FIXED. CNN은 lr·dropout이 필수라 FIXED를 기본값으로 둔다.
        self.params = params or CNN1D_FIXED_PARAMS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.channel_mean_ = None
        self.channel_std_ = None
        self.history_ = []

    def _build_model(self):
        return CNN1DNet(
            in_channels=self.params['in_channels'],
            channels=self.params['channels'],
            wide_kernel=self.params['wide_kernel'],
            dropout=self.params['dropout'],
        ).to(self.device)

    def _fit_normalizer(self, X):
        # X: (N, C, L) — 채널별 평균/표준편차 (SVM의 StandardScaler.fit에 해당)
        self.channel_mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.channel_std_ = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.channel_std_ = np.where(self.channel_std_ < 1e-8, 1.0, self.channel_std_)

    def _normalize(self, X):
        return ((X - self.channel_mean_) / self.channel_std_).astype(np.float32)

    def _make_loader(self, X, y, shuffle):
        tx = torch.from_numpy(X)
        ty = torch.from_numpy(y.astype(np.float32))
        ds = TensorDataset(tx, ty)
        return DataLoader(
            ds,
            batch_size=self.params['batch_size'],
            shuffle=shuffle,
        )

    def train(self, X_train, y_train):
        """
        Train 세트만으로 학습한다. (SVM/RF의 train과 동일)
        Validation은 run_cnn1d_622에서 학습 후 predict로만 사용한다.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_train.ndim != 3:
            raise ValueError(f"X_train은 (N, C, L) 이어야 합니다. got {X_train.shape}")
        if X_train.shape[1] != self.params['in_channels']:
            raise ValueError(
                f"채널 수 불일치: X_train C={X_train.shape[1]}, "
                f"in_channels={self.params['in_channels']}"
            )
        if X_train.shape[2] != self.params['signal_length']:
            raise ValueError(
                f"길이 불일치: X_train L={X_train.shape[2]}, "
                f"signal_length={self.params['signal_length']}"
            )
        if len(X_train) != len(y_train):
            raise ValueError("X_train과 y_train 길이가 다릅니다.")

        _set_seed(self.params.get('random_state', RANDOM_STATE))
        self._fit_normalizer(X_train)
        X_tr = self._normalize(X_train)
        y_tr = y_train

        train_loader = self._make_loader(X_tr, y_tr, shuffle=True)

        self.model = self._build_model()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay'],
        )
        criterion = nn.BCEWithLogitsLoss()
        self.history_ = []

        for epoch in range(1, self.params['epochs'] + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.history_.append({'epoch': epoch, 'train_loss': avg_loss})
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f}")

        return self

    @torch.no_grad()
    def _predict_loader(self, loader):
        self.model.eval()
        preds, trues = [], []
        for xb, yb in loader:
            xb = xb.to(self.device)
            logits = self.model(xb)
            pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy().astype(np.int64))
        return np.concatenate(trues), np.concatenate(preds)

    def predict(self, X):
        if self.model is None or self.channel_mean_ is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X는 (N, C, L) 이어야 합니다. got {X.shape}")
        if X.shape[1] != self.params['in_channels']:
            raise ValueError(
                f"채널 수 불일치: X C={X.shape[1]}, "
                f"in_channels={self.params['in_channels']}"
            )
        X_n = self._normalize(X)
        loader = self._make_loader(X_n, np.zeros(len(X), dtype=np.float32), shuffle=False)
        _, y_pred = self._predict_loader(loader)
        return y_pred

    def evaluate(self, X, y, set_name='Test'):
        y_pred = self.predict(X)
        print(f"\n 1D-CNN - {set_name} 평가")
        metrics = print_evaluation_report(y, y_pred)
        return y_pred, metrics


def run_cnn1d_622(X_train, X_val, X_test, y_train, y_val, y_test):
    print("=" * 60)
    print("1D-CNN (WDCNN-style) - 6:2:2 홀드아웃 검증")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"입력 shape: {X_train.shape[1:]}")

    param_keys = list(CNN1D_PARAM_GRID.keys())
    param_values = list(CNN1D_PARAM_GRID.values())

    best_f1 = -1.0
    best_params = None
    search_log = []

    # SVM/RF와 동일: Train만 학습 → Val F1로 조합 비교
    for combo in itertools.product(*param_values):
        trial_params = dict(zip(param_keys, combo))
        merged_params = {**CNN1D_BASE_PARAMS, **trial_params}
        print(f"\n> Trial: {trial_params}")

        clf = CNN1DClassifier(params=merged_params)
        clf.train(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

        search_log.append({'params': trial_params, 'val_f1': val_f1})
        print(f"  -> Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = trial_params

    print(f"\n총 {len(search_log)}개 조합 탐색 완료")
    print(f"{'lr':>10} | {'dropout':>8} | {'Val F1':>8}")
    print("-" * 34)
    for log in search_log:
        p = log['params']
        marker = " *" if p == best_params else ""
        print(f"{p['lr']:>10.0e} | {p['dropout']:>8.1f} | {log['val_f1']:>8.4f}{marker}")
    print(f"\n최적 파라미터: {best_params} (Val F1: {best_f1:.4f})")

    # 최적 파라미터로 최종 모델 학습 및 Val/Test 평가 (SVM/RF와 동일)
    final_params = {**CNN1D_BASE_PARAMS, **best_params}
    final_clf = CNN1DClassifier(params=final_params)
    print("\n최적 파라미터로 재학습...")
    final_clf.train(X_train, y_train)

    y_val_pred, val_metrics = final_clf.evaluate(X_val, y_val, set_name='Validation')
    y_test_pred, test_metrics = final_clf.evaluate(X_test, y_test, set_name='Test')

    return {
        'model': final_clf,
        'best_params': best_params,
        'search_log': search_log,
        'val': {'y_true': y_val, 'y_pred': y_val_pred, 'metrics': val_metrics},
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
    }


def run_cnn1d_73(X_train, X_test, y_train, y_test):
    print("=" * 60)
    print("1D-CNN (WDCNN-style) - 7:3 분할 (고정 파라미터)")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"입력 shape: {X_train.shape[1:]}")

    clf = CNN1DClassifier(params=CNN1D_FIXED_PARAMS)
    clf.train(X_train, y_train)
    y_test_pred, test_metrics = clf.evaluate(X_test, y_test, set_name='Test')

    return {
        'model': clf,
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'metrics': test_metrics},
    }
