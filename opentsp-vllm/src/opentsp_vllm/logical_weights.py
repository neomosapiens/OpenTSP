"""Shared on-disk tensor names for the GPT-2 family used by serving."""
from safetensors.numpy import load, save, save_file


def arrays(model, layer_names):
    model.validate()
    tensors = {name: getattr(model, name) for name in ("wte", "wpe", "lnf_g", "lnf_b")}
    for index, layer in enumerate(model.layers):
        tensors.update({f"layer.{index}.{key}": layer[key] for key in layer_names})
    return tensors


def encode(model, layer_names, format_id):
    return save(arrays(model, layer_names), metadata={"format": format_id})


def write(path, model, layer_names, format_id):
    save_file(arrays(model, layer_names), path, metadata={"format": format_id})


def decode(data, cfg, factory, layer_names):
    tensors = load(data)
    try:
        base = {name: tensors.pop(name) for name in ("wte", "wpe", "lnf_g", "lnf_b")}
        layers = [{key: tensors.pop(f"layer.{i}.{key}") for key in layer_names}
                  for i in range(cfg.layers)]
        if tensors:
            raise ValueError("unexpected tensors in shared model weights")
        return factory(cfg, layers=layers, **base).validate()
    except (KeyError, AssertionError) as error:
        raise ValueError("invalid shared model tensor shape, dtype or name") from error
