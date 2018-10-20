# coding; utf-8


# layersリストの中にさらにリストがあっても、ネットワークがつながる。
def inst_layers(layers, in_layer):
    x = in_layer
    for layer in layers:
        if isinstance(layer, list):
            x = inst_layers(layer, x)
        else:
            x = layer(x)

    return x
