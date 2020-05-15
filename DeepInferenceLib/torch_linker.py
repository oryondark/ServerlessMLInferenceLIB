
def tool_check(toolName, layer):
    if toolName == "torch":
        return torchToTheano

'''
getattr function will be return value of attribute if the class has it.
'''
def attr_parse(layer):
    # You can set a attribute below list box.
    # If the class has not any attribute, may return null
    param_list = [
        'stride',
        'kernel_size',
        'padding',
    ]

    info = {}
    for p in param_list:
        value = getattr(layer, p, False)
        if value == False:
            pass
        else:
            info[str(p)] = value

    info["name"] = layer._get_name()
    return info
