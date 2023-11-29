#!/usr/bin/env python
# 
# This file is part of the imwip_torch distribution (https://github.com/snipdome/imwip_torch).
# Copyright (c) 2022-2023 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Author: Domenico Iuso
# Date: 29/11/2023

import torch
import numba
import imwip

    
class Warp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, u, v, w=None):
        device='cuda'
        if image.device == torch.device('cpu'): # send to gpu
            uu, vv, ww = u.detach().cuda().contiguous(), v.detach().cuda().contiguous(), w.detach().cuda().contiguous() if w is not None else None
            ii = image.detach().cuda().contiguous()
        else:
            uu, vv, ww = u.detach().contiguous(), v.detach().contiguous(), w.detach().contiguous() if w is not None else None
            ii = image.detach().contiguous()
        ctx.u, ctx.v, ctx.w, ctx.image = uu, vv, ww, ii
        uu = numba.cuda.as_cuda_array(ctx.u)
        vv = numba.cuda.as_cuda_array(ctx.v)
        ww = numba.cuda.as_cuda_array(ctx.w) if ctx.w is not None else None
        ii = numba.cuda.as_cuda_array(ctx.image)
        with torch.no_grad():
            output = torch.zeros(image.shape, device=device, dtype=torch.float32, requires_grad=None).contiguous()
            out = numba.cuda.as_cuda_array(output)
            if w is None:
                imwip.warp(ii, uu, vv, degree=3, backend='numba', out=out)
            else:
                imwip.warp(ii, uu, vv, ww, degree=3, backend='numba', out=out)
            output.requires_grad = u.requires_grad or v.requires_grad or (w is not None and w.requires_grad)
        return output.to(image.device)
        
    @staticmethod
    def backward(ctx, grad_output):
        device = 'cuda'
        dfdx = torch.zeros(grad_output.shape, device=device, requires_grad=False, dtype=torch.float32).contiguous()
        dfdy = torch.zeros(grad_output.shape, device=device, requires_grad=False, dtype=torch.float32).contiguous()
        dfdz = torch.zeros(grad_output.shape, device=device, requires_grad=False, dtype=torch.float32).contiguous()
        with torch.no_grad():
            diff_x = numba.cuda.as_cuda_array(dfdx)
            diff_y = numba.cuda.as_cuda_array(dfdy)
            diff_z = numba.cuda.as_cuda_array(dfdz)
            uu = numba.cuda.as_cuda_array(ctx.u)
            vv = numba.cuda.as_cuda_array(ctx.v)
            ww = numba.cuda.as_cuda_array(ctx.w) if ctx.w is not None else None
            ii = numba.cuda.as_cuda_array(ctx.image)
            if ctx.w is None:
                imwip.diff_warp(ii, uu, vv, backend='numba', diff_x=diff_x, diff_y=diff_y, diff_z=diff_z)
            else:
                imwip.diff_warp(ii, uu, vv, ww, backend='numba', diff_x=diff_x, diff_y=diff_y, diff_z=diff_z)
            dfdx = dfdx.to(device=grad_output.device)*grad_output # they should inherit the requires_grad flag from grad_output
            dfdy = dfdy.to(device=grad_output.device)*grad_output
            dfdz = dfdz.to(device=grad_output.device)*grad_output if ctx.w is not None else None
        return None, dfdx, dfdy, dfdz
    