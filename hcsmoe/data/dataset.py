import torch

class CacheDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.alphas = []
        self.Xs = []
        self.Zs = []
        self.Rs = []
        self.prepared = False

    def __len__(self):
        if not self.prepared:
            self.prepare_for_loader()
        return len(self.alphas)

    def __getitem__(self, index):
        if not self.prepared:
            self.prepare_for_loader()
        if isinstance(index, list):
            return [(self.alphas[idx], self.Xs[idx], self.Zs[idx], self.Rs[idx]) for idx in index]
        elif isinstance(index, int):
            return self.alphas[index], self.Xs[index], self.Zs[index], self.Rs[index]

    def append(self, alpha=None, X=None, Z=None, R=None):
        if alpha is not None:
            self.alphas.append(alpha.detach().to('cpu', non_blocking=True))
        if X is not None:
            self.Xs.append(X.detach().to('cpu', non_blocking=True))
        if Z is not None:
            self.Zs.append(Z.detach().to('cpu', non_blocking=True))
        if R is not None:
            self.Rs.append(R.detach().to('cpu', non_blocking=True))
        self.prepared = False

    def prepare_for_loader(self):
        if self.prepared:
            return
        self.prepared = True
        if len(self.alphas) != 0:
            self.alphas = torch.concat(self.alphas)
        if len(self.Xs) != 0:
            self.Xs = torch.concat(self.Xs)
        if len(self.Zs) != 0:
            self.Zs = torch.concat(self.Zs)
        if len(self.Rs) != 0:
            self.Rs = torch.concat(self.Rs)
        assert len(self.Xs) == len(self.Zs)
