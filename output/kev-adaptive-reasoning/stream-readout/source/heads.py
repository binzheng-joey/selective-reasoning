"""Causal fixed-query pooling; parameters shared across prefix positions."""
import math
import torch
from torch import nn


class Readout(nn.Module):
    def __init__(self, kind, dim=2560, classes=4):
        super().__init__()
        self.kind=kind
        self.linear=nn.Linear(dim,classes,bias=False)
        if kind=='stream_attention':
            self.query=nn.Parameter(torch.zeros(dim))

    def forward(self, states, lengths, counts):
        if self.kind=='stream_attention':
            # states [B,L,D]; lengths [B,K], in tokens, not zero-based offsets.
            h=states.float()
            score=torch.einsum('bld,d->bl',h,self.query)/math.sqrt(h.shape[-1])
            mask=torch.arange(h.shape[1],device=h.device)[None,None,:]>=lengths[:,:,None]
            weights=score[:,None,:].expand(-1,lengths.shape[1],-1).masked_fill(mask,-torch.inf).softmax(-1)
            # Equivalent to a linear head after attention pooling, but avoids B*K*L*D storage.
            z=torch.bmm(weights,self.linear(h))
        else:
            z=self.linear(states.float())
        invalid=torch.arange(z.shape[-1],device=z.device)[None,None,:]>=counts[:,None,None]
        return z.masked_fill(invalid,-torch.inf)

    @torch.no_grad()
    def stream(self, sequence):
        """All prefix logits with a constant-size recurrent numerator and denominator."""
        assert self.kind=='stream_attention'
        z=self.linear(sequence.float())
        scores=sequence.float()@self.query/math.sqrt(sequence.shape[-1])
        top=torch.tensor(-torch.inf,device=z.device)
        den=torch.zeros((),device=z.device)
        num=torch.zeros(z.shape[-1],device=z.device)
        out=[]
        for score,value in zip(scores,z):
            updated=torch.maximum(top,score)
            old=torch.exp(top-updated)
            weight=torch.exp(score-updated)
            num=num*old+weight*value
            den=den*old+weight
            top=updated
            out.append(num/den)
        return torch.stack(out)


def check():
    torch.manual_seed(17)
    h=torch.randn(2,19,11)
    lengths=torch.tensor([[1,5,19],[2,8,17]])
    counts=torch.tensor([3,4])
    model=Readout('stream_attention',dim=11)
    with torch.no_grad():model.query.normal_()
    full=model(h,lengths,counts)
    for b in range(2):
        stream=model.stream(h[b])
        assert torch.allclose(full[b,:,:counts[b]],stream[lengths[b]-1,:counts[b]],atol=2e-6,rtol=2e-6)
        for j,n in enumerate(lengths[b]):
            prefix=model(h[b:b+1,:n],n.reshape(1,1),counts[b:b+1])
            assert torch.allclose(prefix[0,0,:counts[b]],full[b,j,:counts[b]],atol=2e-6,rtol=2e-6)
    altered=h.clone();altered[:,10:]=1000*torch.randn_like(altered[:,10:])
    short=torch.tensor([[1,5],[2,8]])
    assert torch.equal(model(h,short,counts),model(altered,short,counts))
    # At initialization attention is exactly uniform: same head weights => mean baseline.
    with torch.no_grad():model.query.zero_()
    uniform=model(h,lengths,counts)
    for b in range(2):
        for j,n in enumerate(lengths[b]):
            expected=model.linear(h[b,:n].mean(0))
            assert torch.allclose(uniform[b,j,:counts[b]],expected[:counts[b]],atol=1e-6)
    # Check gradient reaches both answer mapping and attention query.
    fresh=model(h,lengths,counts)
    loss=torch.nn.functional.cross_entropy(fresh.reshape(-1,4),torch.tensor([0,1,2,1,2,3]))
    loss.backward()
    assert model.linear.weight.grad is not None and model.query.grad is not None
    assert torch.isfinite(model.query.grad).all()
    return dict(causal_mask=True,stream_batch_equivalence=True,uniform_mean_equivalence=True,gradient=True)


if __name__=='__main__':print(check())
