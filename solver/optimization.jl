# Computes the norm of the gradients
function grad_norms(grads, params, p=2)
    gnorm = 0
    for param in params
        gnorm += sum(grads[param].data .^ p)
    end
    gnorm ^ (1. / p)
end

# Clips the norm of the gradient to max norm
# Returns the scaling factor to multiply the gradient to
function clip_norm_scale(grads, params, max_norm, p=2)
    gnorm = grad_norms(grads, params, p)
    gnorm > max_norm ? max_norm/gnorm : 1
end

# Returns the clipped value of the gradient norms
# used for diagnosing
clipped_grad_norms(grads, params, max_norm, p=2) = min(max_norm, grad_norms(grads, params, p))

# update the parameters with gradient clipping
function update_with_clip!(opt, grads, params, max_norm, p=2)
    scale = clip_norm_scale(grads, params, max_norm, p)
    for p in params
        update!(opt, p, scale*grads[p])
    end
end