def freeze_except_qkv(model):
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, type(model.pretrained_model.model.layers[0].self_attn)):
            # Unfreeze Q, K, V projection layers
            module.q_proj.weight.requires_grad = True
            module.k_proj.weight.requires_grad = True
            module.v_proj.weight.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"training model with {trainable_params:,} out of {total_params:,} parameters, i.e. {100 * trainable_params / total_params:.2f}%")
