import time

import tensor_comprehensions as tc
import tensor_comprehensions.tc_unit as tcu
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function


class BatchReNorm2dTCFunction(Function):
    LANG = """
def calc_mean_std(float(N,C,H,W) I, float(6) params) 
-> (batchMean, batchStd)
{
   batchMean(c) +=! I(nn, c, hh, ww)
   batchMean(c) = batchMean(c) / (N * H * W)
   
   batchStd(c) +=! (I(nn, c, hh, ww) - batchMean(c)) * (I(nn, c, hh, ww) - batchMean(c))
   batchStd(c) = sqrt(batchStd(c) / (N * H * W) + params(2))
}

def calc_r_d(float(C) batchStd, float(C) batchMean, float(C) rMeanIn, float(C) rStdIn, float(6) params)
-> (r, d)
{
   r(c) = batchStd(c) / rStdIn(c)
   r(c) = fmin(params(3), fmax(params(4), r(c)))
   d(c) = (batchMean(c) - rMeanIn(c)) / rStdIn(c)
   d(c) = fmin(params(5), fmax(-params(5), d(c)))
}

def calc_O(float(N,C,H,W) I, float(C) weight, float(C) bias, float(C) batchStd, float(C) batchMean, float(C) r, float(C) d)
-> (O)
{
   O(n, c, h, w) = (I(n, c, h, w) - batchMean(c)) / batchStd(c) * r(c) + d(c)
   O(n, c, h, w) = weight(c) * O(n, c, h, w) + bias(c)
}

def calc_running_mean_std(float(C) batchStd, float(C) batchMean, float(C) rMeanIn, float(C) rStdIn, float(6) params)
-> (rMeanOut, rStdOut)
{
   rMeanOut(c) = params(1) * rMeanIn(c) + params(0) * batchMean(c)
   rStdOut(c) = params(1) * rStdIn(c) + params(0) * batchStd(c)
}


def batch_renorm(float(N,C,H,W) I, float(C) rMeanIn, float(C) rStdIn, float(C) weight, float(C) bias, float(6) params)
-> (O, rMeanOut, rStdOut, batchMean, batchStd, r, d)
{
   batchMean(c) +=! I(nn, c, hh, ww)
   batchMean(c) = batchMean(c) / (N * H * W)
   
   batchStd(c) +=! (I(nn, c, hh, ww) - batchMean(c)) * (I(nn, c, hh, ww) - batchMean(c))
   batchStd(c) = sqrt(batchStd(c) / (N * H * W) + params(2))
   
   r(c) = batchStd(c) / rStdIn(c)
   r(c) = fmin(params(3), fmax(params(4), r(c)))
   d(c) = (batchMean(c) - rMeanIn(c)) / rStdIn(c)
   d(c) = fmin(params(5), fmax(-params(5), d(c)))

   O(n, c, h, w) = (I(n, c, h, w) - batchMean(c)) / batchStd(c) * r(c) + d(c)
   O(n, c, h, w) = weight(c) * O(n, c, h, w) + bias(c)

   rMeanOut(c) = params(1) * rMeanIn(c) + params(0) * batchMean(c)
   rStdOut(c) = params(1) * rStdIn(c) + params(0) * batchStd(c)
}

def calc_xHat_grad(float(C) weight, float(N,C,H,W) O_grad)
-> (xHat_grad)
{
    xHat_grad(nn, c, hh, ww) = O_grad(nn, c, hh, ww) * weight(c)
}

def calc_mean_std_grad(float(N,C,H,W) I, float(C) batchMean, float(C) batchStd, float(C) r, float(N,C,H,W) xHat_grad)
-> (batchMean_grad, batchStd_grad)
{
    batchStd_grad(c) +=! xHat_grad(nn, c, hh, ww) * (I(nn, c, hh, ww) - batchMean(c))
    batchStd_grad(c) = batchStd_grad(c) * -r(c) / (batchStd(c) * batchStd(c))
    batchMean_grad(c) +=! xHat_grad(nn, c, hh, ww)
    batchMean_grad(c) = batchMean_grad(c) * -r(c) / batchStd(c)
}

def calc_xHat(float(N,C,H,W) I, float(C) batchMean, float(C) batchStd, float(C) r, float(C) d)
-> (xHat)
{
    xHat(n, c, h, w) = (I(n, c, h, w) - batchMean(c)) / batchStd(c) * r(c) + d(c)
}

def calc_weight_bias_grad(float(N,C,H,W) O_grad, float(N,C,H,W) xHat)
-> (weight_grad, bias_grad)
{
    weight_grad(c) +=! O_grad(nn, c, hh, ww) * xHat(nn, c, hh, ww)
    bias_grad(c) +=! O_grad(nn, c, hh, ww)
}

def calc_I_grad(float(N,C,H,W) I, float(C) batchMean, float(C) batchStd, float(C) r, float(N,C,H,W) xHat_grad, float(C) batchMean_grad, float(C) batchStd_grad)
-> (I_grad)
{
    I_grad(n, c, h, w) = xHat_grad(n, c, h, w) * r(c) / batchStd(c) + batchStd_grad(c) * (I(n, c, h, w) - batchMean(c)) / (batchStd(c) * N * H * W) + batchMean_grad(c) * (1 / (N * H * W))
}

def batch_renorm_grad(float(N,C,H,W) I, float(C) weight, float(C) batchMean, float(C) batchStd, float(C) r, float(C) d, float(N,C,H,W) O_grad) 
-> (I_grad, weight_grad, bias_grad, batchMean_grad, batchStd_grad, xHat_grad, xHat) 
{
    xHat_grad(nn, c, hh, ww) = O_grad(nn, c, hh, ww) * weight(c)
    batchStd_grad(c) +=! xHat_grad(nn, c, hh, ww) * (I(nn, c, hh, ww) - batchMean(c))
    batchStd_grad(c) = batchStd_grad(c) * -r(c) / (batchStd(c) * batchStd(c))
    batchMean_grad(c) +=! xHat_grad(nn, c, hh, ww)
    batchMean_grad(c) = batchMean_grad(c) * -r(c) / batchStd(c)
    xHat(n, c, h, w) = (I(n, c, h, w) - batchMean(c)) / batchStd(c) * r(c) + d(c)
    weight_grad(c) +=! O_grad(nn, c, hh, ww) * xHat(nn, c, hh, ww)
    bias_grad(c) +=! O_grad(nn, c, hh, ww)
    I_grad(n, c, h, w) = xHat_grad(n, c, h, w) * r(c) / batchStd(c) + batchStd_grad(c) * (I(n, c, h, w) - batchMean(c)) / (batchStd(c) * N * H * W) + batchMean_grad(c) * (1 / (N * H * W))
}
    """

    calc_mean_std = tc.define(LANG, name="calc_mean_std")
    calc_r_d = tc.define(LANG, name="calc_r_d")
    calc_O = tc.define(LANG, name="calc_O")
    calc_running_mean_std = tc.define(LANG, name="calc_running_mean_std")
    calc_xHat_grad = tc.define(LANG, name="calc_xHat_grad")
    calc_mean_std_grad = tc.define(LANG, name="calc_mean_std_grad")
    calc_xHat = tc.define(LANG, name="calc_xHat")
    calc_weight_bias_grad = tc.define(LANG, name="calc_weight_bias_grad")
    calc_I_grad = tc.define(LANG, name="calc_I_grad")

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, running_mean, running_std, weight, bias,
                training, momentum, eps, rmax, dmax):
        ctx.save_for_backward(input, weight)

        params = input.new([momentum, 1 - momentum, eps, rmax, 1 / rmax, dmax])

        batchMean, batchStd = BatchReNorm2dTCFunction.calc_mean_std(input, params)
        r, d = BatchReNorm2dTCFunction.calc_r_d(batchStd, batchMean, running_mean, running_std, params)
        O = BatchReNorm2dTCFunction.calc_O(input, weight, bias, batchStd, batchMean, r, d)
        rMeanOut, rStdOut = BatchReNorm2dTCFunction.calc_running_mean_std(batchStd, batchMean, running_mean, running_std, params)

        O, rMeanOut, rStdOut, batchMean, batchStd, r, d = \
            [v.data for v in (O, rMeanOut, rStdOut, batchMean, batchStd, r, d)]
        ctx.batchMean = batchMean
        ctx.batchStd = batchStd
        ctx.r = r
        ctx.d = d

        if training:
            running_mean.copy_(rMeanOut)
            running_std.copy_(rStdOut)

        return O

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight = ctx.saved_variables
        batchMean = ctx.batchMean
        batchStd = ctx.batchStd
        r = ctx.r
        d = ctx.d

        input, weight, grad_output = input.data, weight.data, grad_output.data

        xHat_grad = BatchReNorm2dTCFunction.calc_xHat_grad(weight, grad_output)
        batchMean_grad, batchStd_grad = BatchReNorm2dTCFunction.calc_mean_std_grad(input, batchMean, batchStd, r, xHat_grad)
        xHat = BatchReNorm2dTCFunction.calc_xHat(input, batchMean, batchStd, r, d)
        weight_grad, bias_grad = BatchReNorm2dTCFunction.calc_weight_bias_grad(grad_output, xHat)
        I_grad = BatchReNorm2dTCFunction.calc_I_grad(input, batchMean, batchStd, r, xHat_grad, batchMean_grad, batchStd_grad)

        return I_grad, None, None, weight_grad, bias_grad, None, None, None, None, None


class BatchReNorm2dPTFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, running_mean, running_std, weight, bias,
                training, momentum, eps, rmax, dmax):
        assert training and weight is not None and bias is not None
        if training:
            # (C, B * H * W)
            input_1d = input.transpose(0, 1).contiguous().view(input.shape[1], -1)
            sample_mean = input_1d.mean(1)
            sample_std = (input_1d.var(1) + eps).sqrt()

            r = torch.clamp(sample_std / running_std, 1. / rmax, rmax)
            d = torch.clamp((sample_mean - running_mean) / running_std, -dmax, dmax)

            input_normalized = (input - sample_mean.view(1, -1, 1, 1)) / sample_std.view(1, -1, 1, 1)
            input_normalized = input_normalized * r.view(1, -1, 1, 1) + d.view(1, -1, 1, 1)

            running_mean += momentum * (sample_mean - running_mean)
            running_std += momentum * (sample_std - running_std)
        else:
            input_normalized = (input - running_mean.view(1, -1, 1, 1)) / running_std.view(1, -1, 1, 1)

        ctx.save_for_backward(input, weight)
        ctx.sample_mean = sample_mean
        ctx.sample_std = sample_std
        ctx.r = r
        ctx.d = d

        if weight is not None:
            return input_normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
        else:
            return input_normalized

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, O_grad):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight = ctx.saved_variables
        batchMean = Variable(ctx.sample_mean)
        batchStd = Variable(ctx.sample_std)
        r = Variable(ctx.r)
        d = Variable(ctx.d)

        batchMean_u, batchStd_u = batchMean.view(1, -1, 1, 1), batchStd.view(1, -1, 1, 1)
        r_u, d_u = r.view(1, -1, 1, 1), d.view(1, -1, 1, 1)
        weight_u = weight.view(1, -1, 1, 1)

        # xHat_grad(nn, c, hh, ww) = O_grad(nn, c, hh, ww) * weight(c)
        xHat_grad = O_grad * weight_u
        # batchStd_grad(c) +=! xHat_grad(nn, c, hh, ww) * (I(nn, c, hh, ww) - batchMean(c))
        input_centered = input - batchMean_u
        batchStd_grad = input_centered.mul(xHat_grad)
        batchStd_grad = batchStd_grad.sum(0).view(batchStd_grad.shape[1], -1).sum(-1)
        # batchStd_grad(c) = batchStd_grad(c) * -r(c) / (batchStd(c) * batchStd(c))
        batchStd_grad.mul_(-r).div_(batchStd).div_(batchStd)
        batchStd_grad_u = batchStd_grad.view(1, -1, 1, 1)
        # batchMean_grad(c) +=! xHat_grad(nn, c, hh, ww)
        batchMean_grad = xHat_grad.sum(0).view(xHat_grad.shape[1], -1).sum(-1)
        # batchMean_grad(c) = batchMean_grad(c) * -r(c) / batchStd(c)
        batchMean_grad.mul_(-r).div_(batchStd)
        batchMean_grad_u = batchMean_grad.view(1, -1, 1, 1)
        # xHat(n, c, h, w) = (I(n, c, h, w) - batchMean(c)) / batchStd(c) * r(c) + d(c)
        xHat = input_centered.div(batchStd_u).mul_(r_u).add_(d_u)
        # weight_grad(c) +=! O_grad(nn, c, hh, ww) * xHat(nn, c, hh, ww)
        weight_grad = xHat.mul_(O_grad).sum(0).view(xHat.shape[1], -1).sum(-1)
        # bias_grad(c) +=! O_grad(nn, c, hh, ww)
        bias_grad = O_grad.sum(0).view(O_grad.shape[1], -1).sum(-1)
        # I_grad(n, c, h, w) = xHat_grad(n, c, h, w) * r(c) / batchStd(c)
        #   + batchStd_grad(c) * (I(n, c, h, w) - batchMean(c)) / (batchStd(c) * N * H * W)
        #   + batchMean_grad(c) * (1 / (N * H * W))
        NHW = input.shape[0] * input.shape[2] * input.shape[3]
        I_grad = xHat_grad.mul_(r_u).div_(batchStd_u)
        I_grad.add_(input_centered.mul_(batchStd_grad_u).div_(batchStd_u).mul_(1 / NHW))
        I_grad.add_(batchMean_grad_u.mul(1 / NHW))

        return I_grad, None, None, weight_grad, bias_grad, None, None, None, None, None


def batch_renorm2d(input, running_mean, running_std, weight=None, bias=None,
                   training=False, momentum=0.01, eps=1e-5, rmax=3.0, dmax=5.0):
    if training:
        # (C, B * H * W)
        input_1d = input.transpose(0, 1).contiguous().view(input.shape[1], -1)
        sample_mean = input_1d.mean(1)
        sample_std = (input_1d.var(1) + eps).sqrt()

        r = torch.clamp(sample_std.data / running_std, 1. / rmax, rmax)
        d = torch.clamp((sample_mean.data - running_mean) / running_std, -dmax, dmax)

        input_normalized = (input - sample_mean.view(1, -1, 1, 1)) / sample_std.view(1, -1, 1, 1)
        input_normalized = input_normalized * Variable(r.view(1, -1, 1, 1)) + Variable(d.view(1, -1, 1, 1))

        running_mean += momentum * (sample_mean.data - running_mean)
        running_std += momentum * (sample_std.data - running_std)
    else:
        input_normalized = (input - running_mean.view(1, -1, 1, 1)) / running_std.view(1, -1, 1, 1)

    if weight is not None:
        return input_normalized * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)
    else:
        return input_normalized


def generate_data():
    B, C, H, W = 2, 256, 32, 32
    input = torch.randn(B, C, H, W).cuda()
    running_mean, running_std = torch.randn(C).cuda(), torch.zeros(C).uniform_(0.01, 3).cuda()
    weight, bias = torch.rand(C).cuda(), 0.1 * torch.randn(C).cuda()
    # momentum, 1 - momentum, eps, rmax, 1 / rmax, dmax
    params = input.new([0.01, 0.99, 1e-5, 3.0, 1 / 3.0, 5.0])
    return input, running_mean, running_std, weight, bias, params


def autotune_with_named_cache(unit, *input_tensors, **tuner_kwargs):
    hash_key = tcu.get_tc_hash_key(unit.kwargs_define['name'], *input_tensors)
    tuner_kwargs['cache'] = f'/tmp/{hash_key}'
    unit.autotune(*input_tensors, **tuner_kwargs)


def autotune():
    input, running_mean, running_std, weight, bias, params = generate_data()
    grad_output = input.clone()
    options = tc.Options("group_conv")
    tuner_kwargs = dict(options=options, cache=True, generations=25, pop_size=100, crossover_rate=80, number_elites=10, **tc.autotuner_settings)

    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_mean_std, input, params, **tuner_kwargs)
    batchMean, batchStd = BatchReNorm2dTCFunction.calc_mean_std(input, params)
    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_r_d, batchStd, batchMean, running_mean, running_std, params, **tuner_kwargs)
    r, d = BatchReNorm2dTCFunction.calc_r_d(batchStd, batchMean, running_mean, running_std, params)
    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_O, input, weight, bias, batchStd, batchMean, r, d, **tuner_kwargs)
    O = BatchReNorm2dTCFunction.calc_O(input, weight, bias, batchStd, batchMean, r, d)
    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_running_mean_std, batchStd, batchMean, running_mean, running_std, params, **tuner_kwargs)
    rMeanOut, rStdOut = BatchReNorm2dTCFunction.calc_running_mean_std(batchStd, batchMean, running_mean, running_std, params)

    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_xHat_grad, weight, grad_output, **tuner_kwargs)
    xHat_grad = BatchReNorm2dTCFunction.calc_xHat_grad(weight, grad_output)
    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_mean_std_grad, input, batchMean, batchStd, r, xHat_grad, **tuner_kwargs)
    batchMean_grad, batchStd_grad = BatchReNorm2dTCFunction.calc_mean_std_grad(input, batchMean, batchStd, r, xHat_grad)
    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_xHat, input, batchMean, batchStd, r, d, **tuner_kwargs)
    xHat = BatchReNorm2dTCFunction.calc_xHat(input, batchMean, batchStd, r, d)
    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_weight_bias_grad, grad_output, xHat, **tuner_kwargs)
    weight_grad, bias_grad = BatchReNorm2dTCFunction.calc_weight_bias_grad(grad_output, xHat)
    autotune_with_named_cache(BatchReNorm2dTCFunction.calc_I_grad, input, batchMean, batchStd, r, xHat_grad, batchMean_grad, batchStd_grad, **tuner_kwargs)
    I_grad = BatchReNorm2dTCFunction.calc_I_grad(input, batchMean, batchStd, r, xHat_grad, batchMean_grad, batchStd_grad)


def profile_norm(function, message, *args):
    input, running_mean, running_std, weight, bias, params = generate_data()
    input = Variable(input)
    weight, bias = nn.Parameter(weight), nn.Parameter(bias)
    iters = 2500
    prewarm_iters = 100

    for _ in range(prewarm_iters):
        function(input, running_mean, running_std, weight, bias, True, 0.01, 1e-5, *args).sum().backward()

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iters):
        function(input, running_mean, running_std, weight, bias, True, 0.01, 1e-5, *args).sum().backward()
        torch.cuda.synchronize()
    print(message, (time.time() - start_time) / iters * 1000, 'ms')


def check_gradients():
    def get_args():
        torch.manual_seed(123)
        input, running_mean, running_std, weight, bias, params = generate_data()
        input = Variable(input, requires_grad=True)
        weight, bias = nn.Parameter(weight), nn.Parameter(bias)
        return input, running_mean, running_std, weight, bias, True, 0.01, 1e-5, 3.0, 5.0

    naive_args = get_args()
    out_naive = batch_renorm2d(*naive_args)
    out_naive.mean().backward()
    tc_args = get_args()
    out_tc = BatchReNorm2dTCFunction.apply(*tc_args)
    out_tc.mean().backward()

    def rmse(a, b):
        return (a - b).pow(2).mean() ** 0.5

    print('Output RMSE:', rmse(out_naive.data, out_tc.data))
    print('Running mean RMSE:', rmse(naive_args[1], tc_args[1]))
    print('Running std RMSE:', rmse(naive_args[2], tc_args[2]))
    print('Input grad RMSE:', rmse(naive_args[0].grad.data, tc_args[0].grad.data))
    print('Weight grad RMSE:', rmse(naive_args[3].grad.data, tc_args[3].grad.data))
    print('Bias grad RMSE:', rmse(naive_args[4].grad.data, tc_args[4].grad.data))


def print_performance():
    profile_norm(F.batch_norm, 'THNN Batch Normalization:')
    profile_norm(batch_renorm2d, 'PyTorch Batch Renormalization:', 3.0, 5.0)
    profile_norm(BatchReNorm2dPTFunction.apply, 'PyTorch Function Batch Renormalization:', 3.0, 5.0)
    profile_norm(BatchReNorm2dTCFunction.apply, 'TC Batch Renormalization:', 3.0, 5.0)


autotune()
check_gradients()
print_performance()
