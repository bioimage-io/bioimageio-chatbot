import pkgutil

def get_builtin_extensions():
    extensions = []
    for module in pkgutil.walk_packages(__path__, __name__ + '.'):
        if module.name.endswith('_extension'):
            ext_module = module.module_finder.find_module(module.name).load_module(module.name)
            exts = ext_module.get_extensions()
            for ext in exts:
                if ext.name in [e.name for e in extensions]:
                    raise ValueError(f"Extension name {ext.name} already exists.")
            extensions.extend(exts)
            
    return extensions
