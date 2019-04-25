# Design

## Plugin architecture

We want people to be able to provide their own flow methods. We
will provide a [namespace package](https://packaging.python.org/guides/packaging-namespace-packages/), e.g. `flowty.methods` that people can
add to. We can then use something like this:

```python
import importlib
import pkgutil

import myapp.plugins

def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

myapp_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(myapp.plugins)
}
```

We will have a subparser for each method, e.g. `flowty tvl1 <src> <dest> --tvl1-arg1`
which each plugin will provide. They'll have to inherit a parent parser that takes care
of the `<src>` and `<dest>` bits, as we'll always need them.




### References

- https://packaging.python.org/guides/creating-and-discovering-plugins/
https://packaging.python.org/guides/packaging-namespace-packages/
- https://packaging.python.org/guides/packaging-namespace-packages/
- https://alysivji.github.io/simple-plugin-system.html
