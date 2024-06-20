module training

export train_step

using Flux
using Optimisers

# Step LR scheduler 
function step_decay(epoch, LR, step, decay, min_LR)
    return max(LR * decay^(epoch // step), min_LR)
end

function train_step(m, opt_state, train_loader, test_loader, loss, epoch)
    train_loss = 0.0
    test_loss = 0.0

    # Training
    for (x, y) in train_loader
        loss_val, grad = Flux.withgradient(model -> loss(model, x, y), m)
        opt_state, m = Optimisers.update(opt_state, m, grad[1])
        train_loss += loss_val
    end

    # Testing
    for (x, y) in test_loader
        test_loss += loss(m, x, y)
    end

    # Update learning rate
    LR = parse(Float32, get(ENV, "LR", "0.01"))
    step = parse(Int, get(ENV, "step", "20"))
    decay = parse(Float32, get(ENV, "decay", "0.8"))
    min_LR = parse(Float32, get(ENV, "min_LR", "1e-4"))
    Optimisers.adjust!(opt_state, step_decay(epoch, LR, step, decay, min_LR))

    train_loss /= length(train_loader.data)
    test_loss /= length(test_loader.data)

    return m, opt_state, train_loss, test_loss
end

end
